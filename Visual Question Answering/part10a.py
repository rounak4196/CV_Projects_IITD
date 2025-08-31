import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BertTokenizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 8
batch_size = 32

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train','inference'])
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--save_path', type=str)
args = parser.parse_args()



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_questions_path = os.path.join(args.dataset, 'questions/CLEVR_trainA_questions.json')
val_questions_path = os.path.join(args.dataset, 'questions/CLEVR_valA_questions.json')
test_questions_path = os.path.join(args.dataset, 'questions/CLEVR_testA_questions.json')
question_image_directory = os.path.join(args.dataset, 'images/trainA')
val_image_directory = os.path.join(args.dataset, 'images/valA')
test_image_directory = os.path.join(args.dataset, 'images/testA') if args.mode == 'inference' else None
testB_image_directory = os.path.join(args.dataset, 'images/testB') if args.mode == 'inference' else None

train_questions = json.load(open(train_questions_path))['questions']
all_answers = ['<PAD>','<UNK>','<CLS>']+(list(set(q['answer'] for q in train_questions)))
answer2idx = {a: i for i, a in enumerate(all_answers)}
num_classes = len(all_answers)
max_len = 1+max(len(tokenizer.encode(q['question'], add_special_tokens=False)) for q in train_questions)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()
    
class CLEVRDataset(Dataset):
    def __init__(self, image_dir, question_file):
        self.questions = json.load(open(question_file))['questions']
        self.image_dir = image_dir
        self.idx2answer = {v: k for k, v in answer2idx.items()}

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        item = self.questions[idx]
        img_path = os.path.join(self.image_dir, item['image_filename'])
        image = Image.open(img_path).convert('RGB')

        image = transform(image)

        question = item['question']
        tokens = tokenizer.encode(
            question,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            add_special_tokens=False
        )
        return (image,torch.tensor(tokens, dtype=torch.long),answer2idx[item['answer']])

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) # removing the final global average pooling and fully connected layers
        self.proj = nn.Conv2d(2048, 768, 1)

        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        return self.proj(x)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim=768, num_layers=6, nhead=8):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))
        # self.position_embedding = nn.Parameter(torch.randn(1, max_len + 1, embed_dim))
        self.position_embedding = nn.Embedding(max_len + 1, embed_dim)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        B, S = x.size()
        padding_mask = (x == 0)  # create padding mask

        # positions with CLS token
        positions = torch.arange(S+1, device=x.device).unsqueeze(0).expand(B, -1)

        _token_embedding = self.token_embedding(x)
        _position_embedding = self.position_embedding(positions)

        cls_embeddings = self.cls_embedding.expand(B, -1, -1)
        embeddings = torch.cat([cls_embeddings, _token_embedding], dim=1)
        embeddings += _position_embedding

        # expand padding mask for CLS token
        padding_mask = torch.cat([torch.zeros(B,1, dtype=torch.bool, device=x.device), padding_mask], dim=1)

        output = self.transformer(embeddings, src_key_padding_mask=padding_mask)
        return output[:, 0]

class FeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=8,
            batch_first=True
        )

    def forward(self, text_features, image_features):
        query = text_features.unsqueeze(1)
        attn_output, _ = self.cross_attn(
            query=query,
            key=image_features,
            value=image_features
        )
        return attn_output.squeeze(1)

class VQAModel(nn.Module):
    def __init__(self, vocab_size, max_len, num_classes):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder(vocab_size, max_len)
        self.fusion = FeatureFusion()
        self.classifier = nn.Sequential(
            nn.Linear(768, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes))

    def forward(self, images, questions):
        img_features = self.image_encoder(images)
        # B, C, h, w = img_features.shape
        img_features = img_features.flatten(2).permute(0,2,1)
        txt_features = self.text_encoder(questions)
        fused = self.fusion(txt_features, img_features)
        return self.classifier(fused)

def evaluate(model:CLEVRDataset, loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, questions, labels in loader:
            images, questions, labels = images.to(device), questions.to(device), labels.to(device)

            outputs = model(images, questions)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1-score: {f1:.4f}')


    correct_indices = [i for i in range(len(all_labels)) if all_preds[i] == all_labels[i]]
    chosen_correct = np.random.choice(correct_indices, 5, replace=False)

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for i, idx in enumerate(chosen_correct):
        image, question, label = test_dataset[idx]
        question_text = tokenizer.decode(question, skip_special_tokens=True)
        image_tensor = image.unsqueeze(0).to(device)
        question_tensor = question.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor, question_tensor)
            pred = output.argmax(dim=1).item()
        axes[i].imshow(image.permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axes[i].axis('off')
        axes[i].set_title(f"Q: {question_text}\nPred: {test_dataset.idx2answer[pred]}\nGT: {test_dataset.idx2answer[label]}")
        print(f"Q: {question_text}\nPred: {test_dataset.idx2answer[pred]}\nGT: {test_dataset.idx2answer[label]}")

    plt.tight_layout()
    plt.suptitle("Correct Predictions", fontsize=16)
    plt.subplots_adjust(top=0.8)
    plt.savefig('correct_predictions.jpg', format='jpg')

    print("---------------------------------------------------------------------------------------")
    wrong_indices = [i for i in range(len(all_labels)) if all_preds[i] != all_labels[i]]
    if len(wrong_indices) >= 5:
        chosen_wrong = np.random.choice(wrong_indices, 5, replace=False)

        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        for i, idx in enumerate(chosen_wrong):
            image, question, label = test_dataset[idx]
            question_text = tokenizer.decode(question, skip_special_tokens=True)
            image_tensor = image.unsqueeze(0).to(device)
            question_tensor = question.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(image_tensor, question_tensor)
                pred = output.argmax(dim=1).item()
            axes[i].imshow(image.permute(1, 2, 0).numpy() * 0.5 + 0.5)
            axes[i].axis('off')
            axes[i].set_title(f"Q: {question_text}\nPred: {test_dataset.idx2answer[pred]}\nGT: {test_dataset.idx2answer[label]}")
            print(f"Q: {question_text}\nPred: {test_dataset.idx2answer[pred]}\nGT: {test_dataset.idx2answer[label]}")

        plt.tight_layout()
        plt.suptitle("Incorrect Predictions (Error Cases)", fontsize=16)
        plt.subplots_adjust(top=0.8)
        plt.savefig('incorrect_predictions.jpg', format='jpg')



if __name__ == '__main__':

    model = VQAModel(tokenizer.vocab_size, max_len, num_classes).to(device)
    # criterion=nn.CrossEntropyLoss()
    criterion = FocalLoss()

    optimizer = optim.Adam([
        {'params': model.text_encoder.parameters()},
        {'params': model.image_encoder.proj.parameters()},
        {'params': model.fusion.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=3e-5)

    if args.mode == 'inference':
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        test_dataset = CLEVRDataset(test_image_directory,test_questions_path,)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
        print("Test Data Loaded!!!!")

        evaluate(model,test_loader)


    else:
        if args.model_path:
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = 4e-6
                
        train_dataset = CLEVRDataset(question_image_directory,train_questions_path,)
        val_dataset = CLEVRDataset(val_image_directory,val_questions_path,)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=8)

        
        best_acc = 0.0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, questions, labels in train_loader:
                images, questions, labels = images.to(device), questions.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images, questions)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / len(train_loader)
            train_acc = correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, questions, labels in val_loader:
                    images, questions, labels = images.to(device), questions.to(device), labels.to(device)

                    outputs = model(images, questions)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            val_loss /= len(val_loader)
            val_acc = correct / total
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accs': train_accs,
                    'val_accs': val_accs,
                    'answer2idx': answer2idx,
                    'max_len': max_len,
                    'vocab_size': tokenizer.vocab_size
                }, args.save_path)


