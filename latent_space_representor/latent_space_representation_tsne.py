import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import models, transforms


from torch.autograd import Variable
from PIL import Image
import os
import torch 

# Load pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
layer = model._modules.get('avgpool')

# Remove the fully connected layer
model = model.eval()
model = model.cuda() if torch.cuda.is_available() else model

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



# Function to extract features from images using ResNet-18
def extract_features_fc(img_path):
    img = Image.open(img_path)
    if len(img.getbands()) == 1:
        #print('###', img_path, np.array(img).shape)
        img = img.convert("RGB")

        #return img
    
    img_tensor = preprocess(img).unsqueeze(0)
    img_variable = Variable(img_tensor)
    img_variable = img_variable.cuda() if torch.cuda.is_available() else img_variable
    features = model(img_variable)
    features = features.squeeze().cpu().detach().numpy()
    return features


def plot2D(plot_title, latent_space, plot_type='t-SNE',notebook=False):
    plt.figure(figsize=(10, 10))
    plt.scatter(x=latent_space[:, 0], y=latent_space[:, 1])

    plt.title(f'{plot_type} Latent Space Visualization {plot_title}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    if notebook:
        plt.show(block=False)
        plt.savefig(plot_title+'_'+plot_type+'2d_.png')
        
        
    else:
         plt.savefig(plot_title+'_'+plot_type+'2d_.png')

def plot3D(plot_title, latent_space, plot_type='t-SNE', notebook=False):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D axis 
    # Scatter plot
    ax.scatter(latent_space[:, 0], latent_space[:, 1], latent_space[:, 2], s=10)

    # Set labels and title
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title(f'{plot_type} Latent Space Visualization {plot_title}')
    if notebook:
        plt.show(block=False)
        plt.savefig(plot_title+'_'+plot_type+'3d_.png')
        
    else:
        plt.savefig(plot_title+'_'+plot_type+'3d_.png')



def extract_features_avgpool(img_path):
    image = Image.open(img_path)
    if len(image.getbands()) == 1:
        #print('###', img_path, np.array(image).shape)
        image = image.convert("RGB")
    # Create a PyTorch tensor with the transformed image
    t_img = preprocess(image)
    # Create a vector of zeros that will hold our feature vector
    # The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)

    # Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.flatten())                 # <-- flatten

    # Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # Run the model on our transformed image
    img_variable = Variable(t_img.unsqueeze(0))
    img_variable = img_variable.cuda() if torch.cuda.is_available() else img_variable
    with torch.no_grad():                               # <-- no_grad context
        model(img_variable)                       # <-- unsqueeze
    # Detach our copy function from the layer
    h.remove()
    return my_embedding.numpy()




def extract_features(mvtec_path, extract_fc=False):
    good_features = []
    anomaly_features = []

    data_dir = os.path.join(mvtec_path, 'train/good')
    for img_file in os.listdir(data_dir):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            img_path = os.path.join(data_dir, img_file)
            if extract_fc:
                img_features = extract_features_fc(img_path)
            else: 
                img_features = extract_features_avgpool(img_path)

            if(img_features is not None):
                good_features.append(img_features)
    data_dir = os.path.join(mvtec_path, 'test')
    anomaly_dir_list = os.listdir(data_dir)
    anomaly_dir_list.remove('good')
    for dir in anomaly_dir_list:
        anomaly_dir = os.path.join(data_dir, dir)
        for img_file in os.listdir(anomaly_dir):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                img_path = os.path.join(anomaly_dir, img_file)
                if extract_fc:
                    img_features = extract_features_fc(img_path)
                else: 
                    img_features = extract_features_avgpool(img_path)
                if(img_features is not None):
                    anomaly_features.append(img_features)

    return good_features, anomaly_features


def list_folders(path):
    folders = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    return folders





mv_tec_dir = '/home/myhome/Downloads/datapath/mvtec'
#mvtec_classes = list_folders(mv_tec_dir)
mvtec_classes = ['capsule', 'bottle', 'carpet', 'leather', 'pill', 'transistor', 'tile', 'cable', 'zipper', 'toothbrush', 'metal_nut', 'hazelnut', 'screw', 'grid', 'wood']
#mv_tec_dir = '/home/myhome/Downloads/datapath'
#mvtec_classes = ['cars', 'road']


mvtec_classes = list_folders(mv_tec_dir)
for mvtec_class in mvtec_classes:
    
    target_dir = os.path.join(mv_tec_dir, mvtec_class)
    good_features, anomaly_features = extract_features(mvtec_path=target_dir)
    combined = good_features + anomaly_features

    good_features = np.array(good_features)
    
    anomaly_features = np.array(anomaly_features)
    
    

    np.save(f'{mvtec_class}_good.npy', good_features)
    np.save(f'{mvtec_class}_anomaly.npy', anomaly_features)
    print(f'{mvtec_class}', good_features.shape, anomaly_features.shape)
    if mvtec_class=='toothbrush' or mvtec_class=='japan' or mvtec_class=='czech':
        tsne2 = TSNE(n_components=2, random_state=42, perplexity=10)
        tsne3 = TSNE(n_components=3, random_state=42, perplexity=10)
    else:
        tsne2 = TSNE(n_components=2, random_state=42)
        tsne3 = TSNE(n_components=3, random_state=42)
    if len(good_features)>0:
        latent_space2 = tsne2.fit_transform(good_features)
        plot2D(f'{mvtec_class.capitalize()} Good', latent_space2)
        latent_space3 = tsne3.fit_transform(good_features)
        plot3D(f'{mvtec_class.capitalize()} Good', latent_space3)

    if len(anomaly_features)>0:
        latent_space2 = tsne2.fit_transform(anomaly_features)
        plot2D(f'{mvtec_class.capitalize()} Anomaly', latent_space2)
        latent_space3 = tsne3.fit_transform(anomaly_features)
        plot3D(f'{mvtec_class.capitalize()} Anomaly', latent_space3)
    if len(good_features)>0 and len(anomaly_features)>0:
        combined = np.array(combined)
        latent_space2 = tsne2.fit_transform(combined)
        plot2D(f'{mvtec_class.capitalize()} Combined Space', latent_space2)
        latent_space3 = tsne3.fit_transform(combined)
        plot3D(f'{mvtec_class.capitalize()} Combined Space', latent_space3)