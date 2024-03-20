import cv2
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Helpers

def create_mel_spectrogram(audio_path):
    # Load the audio file
    waveform, sampling_rate = librosa.load(audio_path, sr=None)

    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sampling_rate)

    # Convert to decibels (log scale)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram_db,sampling_rate

def save_mel_spectrogram_image(audio_path, output_image_path, target_size=(256, 256)):
    # Load the audio file
    mel_spectrogram, sampling_rate = create_mel_spectrogram(audio_path)

    # Create a figure
    fig = plt.figure(frameon=False)
    fig.set_size_inches(target_size[0] / 100, target_size[1] / 100)  # Convert from pixels to inches

    # Add an axis to the figure
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Draw the Mel spectrogram on the axis
    ax.imshow(mel_spectrogram, aspect='auto', cmap='viridis', origin='lower', extent=(0, mel_spectrogram.shape[1], 0, mel_spectrogram.shape[0]))

    # Save the figure as an image file
    fig.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
save_mel_spectrogram_image(audio_file_path, "mel_spectrogram.png")

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# Classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module,PyTorchModelHubMixin):
    def __init__(self, config:dict,  pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(config['image_size'])
        patch_height, patch_width = pair(config['patch_size'])

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, config['dim']),
            nn.LayerNorm(config['dim']),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, config['dim']))
        self.cls_token = nn.Parameter(torch.randn(1, 1, config['dim']))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(config['dim'], config['depth'], config['heads'], dim_head, config['mlp_dim'], dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(config['dim'], config['num_classes'])

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
        
        
config = {
    "image_size": 768,
    "patch_size": 64,
    "num_classes": 1,
    "dim": 1024,
    "depth": 12,
    "heads": 12,
    "mlp_dim": 3072,
    "dropout": 0.2,
    "emb_dropout": 0.2
}

v = ViT(config)

model = v.from_pretrained("LucasVitoriano/AudioAnalyzer")

model.eval()



# Load and preprocess the image
image_path = "mel_spectrogram.png"

img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.einsum('ihj->jhi', torch.from_numpy(img_rgb))
img_tensor = img_tensor.to(torch.float32)

img_tensor = img_tensor.unsqueeze(0)

output = model(img_tensor)
output.item()
