import torch
from src.vocab import SMSvocab
from src.Data_Preprocessing import SMSdataset, prepare_sms_data
from src.embedding import get_embedding_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from src.BiLSTM import SpamAttentionClassifier
# Data Preprocessing ======

BATCH_SIZE = 64
NUM_WORKERS = 2

tr_text,t_text,tr_label,t_label=prepare_sms_data("SMSSpamCollection")
vocab=SMSvocab(min_freq=2)
vocab.build_voabulary(tr_text)
vocab.save_vocab("./Data/vocab.json")
embedding_matrix=get_embedding_matrix("./Data/spam_fasttext_gensim.model",vocab.stoi)
train_ds=SMSdataset(tr_text,tr_label,vocab,max_len=50)
val_ds=SMSdataset(t_text,t_label,vocab,max_len=50)

train_loader = DataLoader(
    dataset=train_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS,
    pin_memory=True  
)

val_loader = DataLoader(
    dataset=val_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# Define Device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train and Validation Loop

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    model.to(device)
    
    # Using BCEWithLogitsLoss for numerical stability
    pos_weight = torch.tensor([7.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Scheduler to help with that final 98% push
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Standard Forward Pass (FP32)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward Pass
            loss.backward()
            
            # --- GRADIENT CLIPPING ---
            # This prevents the 0.5 stagnation by keeping gradients in a healthy range
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        # scheduler.step(val_loss)
        print(f"Epoch {epoch+1} Summary: Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

def evaluate(model, loader, criterion):
    model.eval()
    losses, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            losses += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return losses / len(loader), correct / total

# Init Model and Call training loop

model=SpamAttentionClassifier(embedding_matrix=embedding_matrix)
print("Starting Training========")
try:
    train_model(model,train_loader,val_loader,epochs=15)
except Exception as e:
    print(f"Exception occured{e}")
    raise
print("Training Completed========\nSaving model!")
MODEL_PATH='Spam_Classifier.pth'
torch.save(model.state_dict(),MODEL_PATH)
print(f"Model has beed saved at {MODEL_PATH}")
