"""
Bangladesh River Flood Forecasting - Training Utilities
Model training, evaluation, and metrics calculation functions

Extracted from Phase 3: Steps 10.3, 11.1, and 12.1
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys

def calculate_metrics(predictions, targets, scaler):
    """Calculate evaluation metrics"""
    
    # Convert back to original scale
    pred_original = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    target_original = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = np.mean((pred_original - target_original) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_original - target_original))
    mape = np.mean(np.abs((target_original - pred_original) / (target_original + 1e-6))) * 100
    
    # R-squared
    ss_res = np.sum((target_original - pred_original) ** 2)
    ss_tot = np.sum((target_original - np.mean(target_original)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }


def train_model(model, train_loader, val_loader, adj_matrix, criterion, optimizer, 
                num_epochs, device, model_name):
    """Train spatio-temporal GNN model with enhanced debugging"""
    
    print(f"🔍 DEBUG: Training function called!")
    print(f"🔍 DEBUG: Model name: {model_name}")
    print(f"🔍 DEBUG: Device: {device}")
    print(f"🔍 DEBUG: Number of epochs: {num_epochs}")
    print(f"🔍 DEBUG: Train batches: {len(train_loader)}")
    print(f"🔍 DEBUG: Val batches: {len(val_loader)}")
    
    # Force flush output in notebooks
    sys.stdout.flush()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print(f"\n🚀 Starting {model_name} training...")
    print(f"Device: {device}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print("-" * 60)
    sys.stdout.flush()  # Force output
    
    for epoch in range(num_epochs):
        print(f"🔍 DEBUG: Starting epoch {epoch+1}")
        sys.stdout.flush()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_mae = 0.0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            if batch_idx == 0:  # Print first batch info
                print(f"🔍 DEBUG: Processing first batch, shape: {batch_x.shape}")
                sys.stdout.flush()
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                predictions = model(batch_x, adj_matrix)
                
                # Calculate loss
                total_loss, mse_loss, mae_loss = criterion(predictions, batch_y)
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Accumulate losses
                train_loss += total_loss.item()
                train_mse += mse_loss.item()
                train_mae += mae_loss.item()
                
            except Exception as e:
                print(f"❌ ERROR in batch {batch_idx}: {e}")
                sys.stdout.flush()
                return None, None
        
        # Average training losses
        avg_train_loss = train_loss / len(train_loader)
        avg_train_mse = train_mse / len(train_loader)
        avg_train_mae = train_mae / len(train_loader)
        
        print(f"🔍 DEBUG: Epoch {epoch+1} training completed, avg loss: {avg_train_loss:.6f}")
        sys.stdout.flush()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                try:
                    predictions = model(batch_x, adj_matrix)
                    total_loss, mse_loss, mae_loss = criterion(predictions, batch_y)
                    
                    val_loss += total_loss.item()
                    val_mse += mse_loss.item()
                    val_mae += mae_loss.item()
                except Exception as e:
                    print(f"❌ VALIDATION ERROR: {e}")
                    sys.stdout.flush()
                    return None, None
        
        # Average validation losses
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mse = val_mse / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        
        # Store losses
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Print progress (ALWAYS print first 5 epochs and every 5th epoch)
        if epoch < 5 or (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | "
                  f"Train RMSE: {np.sqrt(avg_train_mse):.4f} | "
                  f"Val RMSE: {np.sqrt(avg_val_mse):.4f}")
            sys.stdout.flush()
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"💾 Saving best model at epoch {epoch+1} with val_loss: {best_val_loss:.6f}")
            sys.stdout.flush()
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f'models/best_{model_name.lower()}_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            sys.stdout.flush()
            break
    
    print(f"\n✅ {model_name} training completed!")
    print(f"🏆 Best validation loss: {best_val_loss:.6f}")
    sys.stdout.flush()
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, adj_matrix, target_scaler, device, model_name):
    """Evaluate model on test set"""
    
    model.eval()
    all_predictions = []
    all_targets = []
    test_losses = []
    
    print(f"\n📊 Evaluating {model_name}...")
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Make predictions
            predictions = model(batch_x, adj_matrix)
            predictions = predictions.squeeze(-1)  # Remove last dimension
            
            # Store predictions and targets
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets, target_scaler)
    
    return all_predictions, all_targets, metrics
