# ============================================================
# DATASET + PREPROCESSING
# QICAN-TamilOCR
# ============================================================

# pip install timm pywavelets albumentations opencv-python

import os
import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from torchvision import transforms

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

IMAGE_SIZE = 224
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset_path = "/content/Dataset"

# Folder Structure
#
# Dataset
#    train
#        class1
#        class2
#    test
#        class1
#        class2

# ------------------------------------------------------------
# Tiny Latent Diffusion Restoration Network (Simplified)
# ------------------------------------------------------------

class TLDRNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2)

        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(64,32,2,stride=2),
            nn.ReLU(),

            nn.Conv2d(32,1,3,padding=1),
            nn.Sigmoid()

        )

    def forward(self,x):

        e=self.encoder(x)
        d=self.decoder(e)

        return d


restoration_model = TLDRNet().to(DEVICE)

# ------------------------------------------------------------
# Wavelet Enhancement
# ------------------------------------------------------------

def wavelet_enhancement(img):

    coeffs = pywt.dwt2(img,'haar')

    LL,(LH,HL,HH)=coeffs

    LH*=1.6
    HL*=1.6
    HH*=2.0

    enhanced = pywt.idwt2((LL,(LH,HL,HH)),'haar')

    enhanced=np.clip(enhanced,0,255)

    return enhanced.astype(np.uint8)

# ------------------------------------------------------------
# Laplacian Pyramid Enhancement
# ------------------------------------------------------------

def laplacian_enhance(img):

    lap=cv2.Laplacian(img,cv2.CV_64F)

    sharp=img+0.7*lap

    sharp=np.clip(sharp,0,255)

    return sharp.astype(np.uint8)

# ------------------------------------------------------------
# Sauvola Threshold
# ------------------------------------------------------------

def sauvola(img,window=25,k=0.2):

    mean=cv2.boxFilter(img,cv2.CV_64F,(window,window))

    sqmean=cv2.boxFilter(img**2,cv2.CV_64F,(window,window))

    std=np.sqrt(sqmean-mean**2)

    R=128

    thresh=mean*(1+k*(std/R-1))

    binary=(img>thresh).astype(np.uint8)*255

    return binary

# ------------------------------------------------------------
# Complete Preprocessing
# ------------------------------------------------------------

def preprocess(path):

    img=cv2.imread(path,0)

    img=cv2.resize(img,(224,224))

    x=torch.tensor(img/255.).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        restored=restoration_model(x)

    restored=restored.squeeze().cpu().numpy()*255

    restored=restored.astype(np.uint8)

    wave=wavelet_enhancement(restored)

    edge=laplacian_enhance(wave)

    binary=sauvola(edge)

    return binary

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class TamilDataset(Dataset):

    def __init__(self,root):

        self.images=[]
        self.labels=[]

        self.classes=sorted(os.listdir(root))

        for i,c in enumerate(self.classes):

            files=glob(os.path.join(root,c,"*"))

            for f in files:

                self.images.append(f)
                self.labels.append(i)

    def __len__(self):

        return len(self.images)

    def __getitem__(self,index):

        img=preprocess(self.images[index])

        img=Image.fromarray(img)

        img=transforms.ToTensor()(img)

        label=self.labels[index]

        return img,label

# ------------------------------------------------------------
# DataLoader
# ------------------------------------------------------------

train_dataset=TamilDataset(dataset_path+"/train")
test_dataset=TamilDataset(dataset_path+"/test")

train_loader=DataLoader(train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True)

test_loader=DataLoader(test_dataset,
                       batch_size=BATCH_SIZE)

print("Training Images :",len(train_dataset))
print("Testing Images :",len(test_dataset))

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

sample=train_dataset.images[0]

original=cv2.imread(sample,0)
processed=preprocess(sample)

plt.figure(figsize=(10,5))

plt.subplot(121)
plt.imshow(original,cmap='gray')
plt.title("Original")

plt.subplot(122)
plt.imshow(processed,cmap='gray')
plt.title("Preprocessed")

plt.show()
# ============================================================
# PART-2 : TamilGlyph Tokenizer
# ============================================================

# pip install timm

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ------------------------------------------------------------
# Depthwise Separable Convolution
# ------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):

    def __init__(self,in_ch,out_ch):

        super().__init__()

        self.depth = nn.Conv2d(
            in_ch,in_ch,
            kernel_size=3,
            padding=1,
            groups=in_ch)

        self.point = nn.Conv2d(
            in_ch,out_ch,
            kernel_size=1)

        self.bn=nn.BatchNorm2d(out_ch)

        self.act=nn.GELU()

    def forward(self,x):

        x=self.depth(x)
        x=self.point(x)
        x=self.bn(x)

        return self.act(x)

# ------------------------------------------------------------
# TamilGlyph Tokenizer
# ------------------------------------------------------------

class TamilGlyphTokenizer(nn.Module):

    def __init__(self):

        super().__init__()

        self.stage1=DepthwiseSeparableConv(1,32)

        self.stage2=DepthwiseSeparableConv(32,64)

        self.stage3=DepthwiseSeparableConv(64,96)

        self.stage4=DepthwiseSeparableConv(96,128)

    def forward(self,x):

        x=self.stage1(x)
        x=self.stage2(x)
        x=self.stage3(x)
        x=self.stage4(x)

        return x

# ------------------------------------------------------------
# MobileViT-v3 Encoder
# ------------------------------------------------------------

class MobileViTEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.model=timm.create_model(

            "mobilevitv2_100",
            pretrained=True,
            features_only=True

        )

    def forward(self,x):

        return self.model(x)[-1]

# ------------------------------------------------------------
# Dynamic Patch Embedding
# ------------------------------------------------------------

class DynamicPatchEmbedding(nn.Module):

    def __init__(self,
                 in_channels=512,
                 embed_dim=384,
                 patch=4):

        super().__init__()

        self.patch=patch

        self.conv=nn.Conv2d(

            in_channels,
            embed_dim,
            kernel_size=patch,
            stride=patch

        )

    def forward(self,x):

        x=self.conv(x)

        B,C,H,W=x.shape

        tokens=x.flatten(2).transpose(1,2)

        return tokens

# ------------------------------------------------------------
# Positional Embedding
# ------------------------------------------------------------

class PositionEmbedding(nn.Module):

    def __init__(self,
                 max_tokens=4096,
                 dim=384):

        super().__init__()

        self.pos=nn.Parameter(

            torch.randn(
                1,
                max_tokens,
                dim))

    def forward(self,x):

        return x+self.pos[:,:x.shape[1],:]

# ------------------------------------------------------------
# Multi-scale Token Generator
# ------------------------------------------------------------

class MultiScaleTokenGenerator(nn.Module):

    def __init__(self):

        super().__init__()

        self.pool1=nn.AdaptiveAvgPool2d(28)

        self.pool2=nn.AdaptiveAvgPool2d(14)

        self.pool3=nn.AdaptiveAvgPool2d(7)

    def token(self,x):

        B,C,H,W=x.shape

        return x.flatten(2).transpose(1,2)

    def forward(self,x):

        t1=self.token(self.pool1(x))

        t2=self.token(self.pool2(x))

        t3=self.token(self.pool3(x))

        return torch.cat([t1,t2,t3],dim=1)

# ------------------------------------------------------------
# TamilGlyphFormer
# ------------------------------------------------------------

class TamilGlyphFormer(nn.Module):

    def __init__(self):

        super().__init__()

        self.tokenizer=TamilGlyphTokenizer()

        self.mobilevit=MobileViTEncoder()

        self.patch=DynamicPatchEmbedding()

        self.multi=MultiScaleTokenGenerator()

        self.position=PositionEmbedding()

    def forward(self,x):

        # Tokenizer

        x=self.tokenizer(x)

        # convert to 3 channels

        x=x.mean(1,keepdim=True)

        x=x.repeat(1,3,1,1)

        # MobileViT

        feat=self.mobilevit(x)

        # Dynamic Patch

        patch_tokens=self.patch(feat)

        # Multi-scale Tokens

        multi_tokens=self.multi(feat)

        # concatenate

        tokens=torch.cat(

            [patch_tokens,multi_tokens],

            dim=1

        )

        # positional encoding

        tokens=self.position(tokens)

        return tokens

# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------

glyphformer=TamilGlyphFormer().to(DEVICE)

images,labels=next(iter(train_loader))

images=images.to(DEVICE)

tokens=glyphformer(images)

print("Input Shape :",images.shape)

print("Token Shape :",tokens.shape)
# ============================================================
# PART-3 : HTMS Transformer
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Dynamic Stroke Pyramid Encoder (DSPE)
# ------------------------------------------------------------

class DSPE(nn.Module):

    def __init__(self, dim=384):

        super().__init__()

        self.pool1 = nn.AvgPool1d(2,2)
        self.pool2 = nn.AvgPool1d(4,4)

        self.fuse = nn.Linear(dim*3, dim)

    def forward(self,x):

        # x : B N C

        p1 = self.pool1(x.transpose(1,2)).transpose(1,2)
        p2 = self.pool2(x.transpose(1,2)).transpose(1,2)

        p1 = F.interpolate(
            p1.transpose(1,2),
            size=x.shape[1],
            mode="nearest"
        ).transpose(1,2)

        p2 = F.interpolate(
            p2.transpose(1,2),
            size=x.shape[1],
            mode="nearest"
        ).transpose(1,2)

        out = torch.cat([x,p1,p2],dim=-1)

        return self.fuse(out)

# ------------------------------------------------------------
# Local Stroke Attention
# ------------------------------------------------------------

class LocalStrokeAttention(nn.Module):

    def __init__(self,
                 dim=384,
                 heads=8):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            dim,
            heads,
            batch_first=True)

    def forward(self,x):

        out,_ = self.attn(x,x,x)

        return out

# ------------------------------------------------------------
# Global Semantic Attention
# ------------------------------------------------------------

class GlobalSemanticAttention(nn.Module):

    def __init__(self,
                 dim=384,
                 heads=8):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            dim,
            heads,
            batch_first=True)

        self.norm = nn.LayerNorm(dim)

    def forward(self,x):

        out,_ = self.attn(x,x,x)

        return self.norm(out+x)

# ------------------------------------------------------------
# Tamil Structural Relation Attention
# ------------------------------------------------------------

class TSRA(nn.Module):

    def __init__(self,
                 dim=384):

        super().__init__()

        self.query = nn.Linear(dim,dim)
        self.key   = nn.Linear(dim,dim)
        self.value = nn.Linear(dim,dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):

        Q=self.query(x)

        K=self.key(x)

        V=self.value(x)

        score=torch.matmul(
            Q,
            K.transpose(-1,-2)
        )/(Q.shape[-1]**0.5)

        attn=self.softmax(score)

        return torch.matmul(attn,V)

# ------------------------------------------------------------
# Dynamic Sparse Token Optimization
# ------------------------------------------------------------

class DSTO(nn.Module):

    def __init__(self,
                 dim=384,
                 keep_ratio=0.7):

        super().__init__()

        self.score = nn.Linear(dim,1)

        self.keep_ratio=keep_ratio

    def forward(self,x):

        importance=self.score(x).squeeze(-1)

        k=max(1,int(
            x.shape[1]*self.keep_ratio))

        idx=torch.topk(
            importance,
            k,
            dim=1
        ).indices

        idx=idx.unsqueeze(-1).expand(
            -1,-1,x.shape[-1])

        tokens=torch.gather(
            x,
            1,
            idx
        )

        return tokens

# ------------------------------------------------------------
# Feed Forward Network
# ------------------------------------------------------------

class FFN(nn.Module):

    def __init__(self,
                 dim=384):

        super().__init__()

        self.net=nn.Sequential(

            nn.Linear(dim,dim*4),

            nn.GELU(),

            nn.Linear(dim*4,dim)

        )

    def forward(self,x):

        return self.net(x)

# ------------------------------------------------------------
# One HTMS Block
# ------------------------------------------------------------

class HTMSBlock(nn.Module):

    def __init__(self,
                 dim=384):

        super().__init__()

        self.dspe = DSPE(dim)

        self.local = LocalStrokeAttention(dim)

        self.global_attn = GlobalSemanticAttention(dim)

        self.tsra = TSRA(dim)

        self.ffn = FFN(dim)

        self.norm1=nn.LayerNorm(dim)

        self.norm2=nn.LayerNorm(dim)

    def forward(self,x):

        x=self.dspe(x)

        x=self.local(x)+x

        x=self.global_attn(x)

        x=self.tsra(x)+x

        x=self.norm1(x)

        x=self.ffn(x)+x

        x=self.norm2(x)

        return x

# ------------------------------------------------------------
# HTMS Transformer
# ------------------------------------------------------------

class HTMSTransformer(nn.Module):

    def __init__(self,
                 depth=6,
                 dim=384):

        super().__init__()

        self.blocks=nn.ModuleList(

            [

                HTMSBlock(dim)

                for _ in range(depth)

            ]

        )

        self.dsto=DSTO(dim)

    def forward(self,x):

        for blk in self.blocks:

            x=blk(x)

        x=self.dsto(x)

        return x

# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------

htms=HTMSTransformer().to(DEVICE)

tokens=glyphformer(images)

features=htms(tokens)

print("Input Tokens :",tokens.shape)

print("HTMS Output :",features.shape)
# ============================================================
# PART-4 : Quantum-Inspired Correlation Attention Module
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Quantum State Representation
# ------------------------------------------------------------

class QuantumState(nn.Module):

    def __init__(self, dim=384):

        super().__init__()

        self.local_proj = nn.Linear(dim, dim)
        self.global_proj = nn.Linear(dim, dim)

    def forward(self, x):

        local = self.local_proj(x)

        global_feat = self.global_proj(x.mean(1, keepdim=True))
        global_feat = global_feat.expand_as(local)

        amplitude = torch.sigmoid(local)

        quantum_state = amplitude * local + (1 - amplitude) * global_feat

        return quantum_state

# ------------------------------------------------------------
# Interference Guided Correlation
# ------------------------------------------------------------

class QuantumCorrelation(nn.Module):

    def __init__(self, dim=384):

        super().__init__()

        self.scale = dim ** -0.5

    def forward(self, q):

        correlation = torch.matmul(
            q,
            q.transpose(-1,-2)
        ) * self.scale

        interference = torch.sin(correlation)

        return correlation + interference

# ------------------------------------------------------------
# Uncertainty Aware Attention
# ------------------------------------------------------------

class UncertaintyAttention(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, score):

        uncertainty = torch.std(score, dim=-1, keepdim=True)

        attention = torch.softmax(
            score/(uncertainty+1e-6),
            dim=-1
        )

        return attention

# ------------------------------------------------------------
# Residual Feature Enhancement
# ------------------------------------------------------------

class ResidualEnhancement(nn.Module):

    def __init__(self, dim=384):

        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(

            nn.Linear(dim, dim*4),

            nn.GELU(),

            nn.Linear(dim*4, dim)

        )

    def forward(self, x, refined):

        out = x + refined

        out = self.norm(out)

        out = self.ffn(out) + out

        return out

# ------------------------------------------------------------
# Complete QICAM
# ------------------------------------------------------------

class QICAM(nn.Module):

    def __init__(self, dim=384):

        super().__init__()

        self.state = QuantumState(dim)

        self.correlation = QuantumCorrelation(dim)

        self.attention = UncertaintyAttention()

        self.refine = ResidualEnhancement(dim)

    def forward(self, x):

        quantum = self.state(x)

        corr = self.correlation(quantum)

        attn = self.attention(corr)

        refined = torch.matmul(attn, quantum)

        out = self.refine(x, refined)

        return out

# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------

qicam = QICAM().to(DEVICE)

tokens = glyphformer(images)

features = htms(tokens)

q_features = qicam(features)

print("HTMS Features :", features.shape)

print("QICAM Output :", q_features.shape)
# ============================================================
# PART-5 : TamilOCR-LLM Adapter
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer
from transformers import AutoModel

# ------------------------------------------------------------
# Feature Projection
# ------------------------------------------------------------

class FeatureProjection(nn.Module):

    def __init__(self,
                 in_dim=384,
                 llm_dim=768):

        super().__init__()

        self.project=nn.Sequential(

            nn.Linear(in_dim,512),

            nn.GELU(),

            nn.Linear(512,llm_dim)

        )

    def forward(self,x):

        return self.project(x)

# ------------------------------------------------------------
# Retrieval Augmented Tamil Lexicon Memory
# ------------------------------------------------------------

class TamilLexiconMemory(nn.Module):

    def __init__(self,
                 llm_dim=768):

        super().__init__()

        self.memory=nn.Parameter(

            torch.randn(
                3000,
                llm_dim
            )
        )

    def forward(self,x):

        score=torch.matmul(

            x,

            self.memory.T

        )

        weight=torch.softmax(

            score,

            dim=-1

        )

        retrieved=torch.matmul(

            weight,

            self.memory

        )

        return retrieved

# ------------------------------------------------------------
# Tiny Language Encoder
# ------------------------------------------------------------

class TinyLanguageEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.model=AutoModel.from_pretrained(

            "google/bert_uncased_L-4_H-512_A-8"

        )

    def forward(self,emb):

        out=self.model(

            inputs_embeds=emb

        )

        return out.last_hidden_state

# ------------------------------------------------------------
# Cross Attention Fusion
# ------------------------------------------------------------

class CrossAttentionFusion(nn.Module):

    def __init__(self,
                 dim=512):

        super().__init__()

        self.attn=nn.MultiheadAttention(

            dim,

            8,

            batch_first=True

        )

    def forward(self,

                visual,

                lexical):

        out,_=self.attn(

            visual,

            lexical,

            lexical

        )

        return out

# ------------------------------------------------------------
# Grammar Refinement
# ------------------------------------------------------------

class GrammarRefinement(nn.Module):

    def __init__(self,
                 dim=512):

        super().__init__()

        self.block=nn.Sequential(

            nn.Linear(dim,dim),

            nn.GELU(),

            nn.Linear(dim,dim)

        )

        self.norm=nn.LayerNorm(dim)

    def forward(self,x):

        return self.norm(

            self.block(x)+x

        )

# ------------------------------------------------------------
# Character Prediction Head
# ------------------------------------------------------------

class OCRHead(nn.Module):

    def __init__(self,
                 dim=512,
                 classes=247):

        super().__init__()

        self.fc=nn.Linear(

            dim,

            classes

        )

    def forward(self,x):

        return self.fc(x)

# ------------------------------------------------------------
# Complete TamilOCR Adapter
# ------------------------------------------------------------

class TamilOCRLLM(nn.Module):

    def __init__(self):

        super().__init__()

        self.project=FeatureProjection()

        self.lexicon=TamilLexiconMemory()

        self.language=TinyLanguageEncoder()

        self.fusion=CrossAttentionFusion()

        self.grammar=GrammarRefinement()

        self.head=OCRHead()

    def forward(self,x):

        visual=self.project(x)

        lexical=self.lexicon(visual)

        language=self.language(visual)

        fused=self.fusion(

            language,

            lexical

        )

        refined=self.grammar(fused)

        logits=self.head(refined)

        return logits,refined

# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------

adapter=TamilOCRLLM().to(DEVICE)

tokens=glyphformer(images)

features=htms(tokens)

features=qicam(features)

logits,language_features=adapter(features)

print("Language Feature :",language_features.shape)

print("Character Logits :",logits.shape)
# ============================================================
# PART-6 : GlyphFormer Decoder
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Cross Attention Fusion
# ------------------------------------------------------------

class CrossAttentionFusion(nn.Module):

    def __init__(self, dim=512):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            batch_first=True
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        out,_ = self.attn(x,x,x)

        return self.norm(out+x)

# ------------------------------------------------------------
# Stroke Reconstruction Layer
# ------------------------------------------------------------

class StrokeReconstruction(nn.Module):

    def __init__(self, dim=512):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(dim,1024),

            nn.GELU(),

            nn.Linear(1024,dim)

        )

    def forward(self,x):

        return self.net(x)+x

# ------------------------------------------------------------
# Edge-aware Upsampling
# ------------------------------------------------------------

class EdgeAwareUpsampling(nn.Module):

    def __init__(self,
                 dim=512,
                 out=512):

        super().__init__()

        self.conv = nn.Sequential(

            nn.ConvTranspose2d(dim,out,2,2),

            nn.BatchNorm2d(out),

            nn.GELU(),

            nn.Conv2d(out,out,3,padding=1),

            nn.BatchNorm2d(out),

            nn.GELU()

        )

    def forward(self,x):

        B,N,C = x.shape

        H = W = int(N**0.5)

        x = x.transpose(1,2)

        x = x.reshape(B,C,H,W)

        x = self.conv(x)

        return x

# ------------------------------------------------------------
# Sparse Decoder Attention
# ------------------------------------------------------------

class SparseDecoderAttention(nn.Module):

    def __init__(self,
                 channels=512):

        super().__init__()

        self.score = nn.Conv2d(
            channels,
            1,
            1
        )

    def forward(self,x):

        attn=torch.sigmoid(

            self.score(x)

        )

        return x*attn

# ------------------------------------------------------------
# Unicode Prediction Head
# ------------------------------------------------------------

class UnicodePrediction(nn.Module):

    def __init__(self,
                 channels=512,
                 classes=247):

        super().__init__()

        self.pool=nn.AdaptiveAvgPool2d(1)

        self.fc=nn.Linear(
            channels,
            classes
        )

    def forward(self,x):

        x=self.pool(x)

        x=x.flatten(1)

        return self.fc(x)

# ------------------------------------------------------------
# GlyphFormer Decoder
# ------------------------------------------------------------

class GlyphFormer(nn.Module):

    def __init__(self):

        super().__init__()

        self.cross = CrossAttentionFusion()

        self.stroke = StrokeReconstruction()

        self.up = EdgeAwareUpsampling()

        self.sparse = SparseDecoderAttention()

        self.head = UnicodePrediction()

    def forward(self,x):

        x=self.cross(x)

        x=self.stroke(x)

        x=self.up(x)

        x=self.sparse(x)

        logits=self.head(x)

        return logits

# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------

decoder = GlyphFormer().to(DEVICE)

tokens = glyphformer(images)

features = htms(tokens)

features = qicam(features)

_,language_features = adapter(features)

prediction = decoder(language_features)

print("Prediction Shape :",prediction.shape)
# ============================================================
# PART-7 : Training + Testing + Evaluation
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Complete Model
# ------------------------------------------------------------

class QICANTamilOCR(nn.Module):

    def __init__(self):

        super().__init__()

        self.tokenizer = TamilGlyphFormer()

        self.htms = HTMSTransformer()

        self.qicam = QICAM()

        self.adapter = TamilOCRLLM()

        self.decoder = GlyphFormer()

    def forward(self,x):

        x = self.tokenizer(x)

        x = self.htms(x)

        x = self.qicam(x)

        _,language = self.adapter(x)

        out = self.decoder(language)

        return out


# ------------------------------------------------------------
# Build Model
# ------------------------------------------------------------

model = QICANTamilOCR().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(

    model.parameters(),

    lr=1e-4,

    weight_decay=1e-4

)

scheduler = optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=20

)

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

EPOCHS = 30

best_acc = 0

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    correct = 0

    total = 0

    for images,labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        pred = outputs.argmax(1)

        correct += (pred==labels).sum().item()

        total += labels.size(0)

    scheduler.step()

    train_acc = correct/total

    print(f"Epoch {epoch+1}")

    print("Loss :",round(train_loss,4))

    print("Accuracy :",round(train_acc*100,2),"%")

    if train_acc>best_acc:

        best_acc=train_acc

        torch.save(

            model.state_dict(),

            "QICAN_TamilOCR_best.pth"

        )

print("Training Completed")

# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------

model.load_state_dict(

    torch.load("QICAN_TamilOCR_best.pth")

)

model.eval()

y_true=[]

y_pred=[]

prob=[]

with torch.no_grad():

    for images,labels in test_loader:

        images=images.to(DEVICE)

        outputs=model(images)

        prediction=outputs.argmax(1)

        y_true.extend(labels.numpy())

        y_pred.extend(prediction.cpu().numpy())

        prob.extend(

            torch.softmax(outputs,1).cpu().numpy()

        )

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

acc=accuracy_score(y_true,y_pred)

pre=precision_score(

    y_true,

    y_pred,

    average="weighted"

)

rec=recall_score(

    y_true,

    y_pred,

    average="weighted"

)

f1=f1_score(

    y_true,

    y_pred,

    average="weighted"

)

print("\nEvaluation Results")

print("----------------------------")

print("Accuracy :",acc)

print("Precision :",pre)

print("Recall :",rec)

print("F1 Score :",f1)

# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------

cm=confusion_matrix(

    y_true,

    y_pred

)

plt.figure(figsize=(8,8))

plt.imshow(cm,cmap="Blues")

plt.colorbar()

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("True")

plt.show()

# ------------------------------------------------------------
# ROC Curve
# ------------------------------------------------------------

if len(np.unique(y_true))==2:

    prob=np.array(prob)

    fpr,tpr,_=roc_curve(

        y_true,

        prob[:,1]

    )

    roc_auc=auc(fpr,tpr)

    plt.figure(figsize=(6,6))

    plt.plot(

        fpr,

        tpr,

        label="AUC = %.4f"%roc_auc

    )

    plt.plot([0,1],[0,1],'--')

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.show()

# ------------------------------------------------------------
# Single Image Prediction
# ------------------------------------------------------------

def predict(image_path):

    model.eval()

    img=preprocess(image_path)

    img=torch.tensor(

        img/255.

    ).float().unsqueeze(0).unsqueeze(0)

    img=img.to(DEVICE)

    with torch.no_grad():

        output=model(img)

    pred=output.argmax(1).item()

    print("Predicted Tamil Character :",pred)

# Example

# predict("/content/test.png")