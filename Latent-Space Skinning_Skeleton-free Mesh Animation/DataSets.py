from torch.utils.data import Dataset
import torch
import numpy as np

class AnimationCategory(Dataset):

    def __init__(self,bones_matrices,mesh_data,Avg_mesh,category_vector):
        self.bones_matrices = bones_matrices
        self.mesh_data = mesh_data
        self.Avg_mesh = Avg_mesh
        self.category_vector = category_vector
        self.withcategories = True

    def __len__(self):
        return 1

    def __getitem__(self,_):
        # print(self.bones_matrices)
        category_tensor = self.category_vector
        category_tensor = torch.tensor(self.category_vector)
        return self.bones_matrices,self.mesh_data,self.Avg_mesh ,category_tensor

class Animation(Dataset):

    def __init__(self,bones_matrices,mesh_data,Avg_mesh):
        self.bones_matrices = bones_matrices
        self.mesh_data = mesh_data
        self.Avg_mesh = Avg_mesh
        self.withcategories = False

    def __len__(self):
        return 1

    def __getitem__(self,_):
        print(self.bones_matrices)
        return self.bones_matrices,self.mesh_data,self.Avg_mesh

class MeshVerticesDataset(Dataset):

    def __init__(self, filename):
        self.data, self.frames, self.vertices, self.values = self.load_data_mesh(filename)
        
    def load_data_mesh(self, filepath):
        all_mesh_frames = np.load(filepath)
        num_frames, num_vertices, num_values = all_mesh_frames.shape
        data = torch.from_numpy(all_mesh_frames)
        data =data.view(num_frames, -1)
        return data,num_frames, num_vertices, num_values
    
    def __len__(self):
        print(len(self.frames), "frames")
        return len(self.frames)
    
    def __getitem__(self, index):  
        frame_mesh = self.data[index]
        return frame_mesh
    
class InputData(Dataset):
    def __init__(self,filename1):
        self.data, self.frames, self.bones, self.values = self.load_data_bones(filename1)
        
    def load_data_bones(self,filename):
        animationBonesMatrices = np.load(filename)
        animationBonesMatrices = animationBonesMatrices[:, :52, :] #kovw se 52 bones
        num_frames,num_bones,num_values = animationBonesMatrices.shape
        data = torch.from_numpy(animationBonesMatrices)
        data=data.view(num_frames, -1)
        return data,num_frames,num_bones,num_values

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self,index):
        frame_bones = self.data[index]
        return frame_bones
    
class CustomDataset(Dataset):

    def __init__(self,List_of_samples):
        self.List_of_samples = List_of_samples
             
    def __len__(self):
        return len(self.List_of_samples)

    def __getitem__(self, idx):
        
        bones = self.List_of_samples[idx].bones_matrices
        mesh_first_frame = self.List_of_samples[idx].mesh_data[0] #check it 
        mesh = self.List_of_samples[idx].mesh_data
        Avg_mesh = self.List_of_samples[idx].Avg_mesh

        if(self.List_of_samples[idx].withcategories == True):
            categories = self.List_of_samples[idx].category_vector
            categories = torch.tensor(categories)
            return bones,mesh_first_frame,mesh,Avg_mesh,categories
        else:
            return bones,mesh_first_frame,mesh,Avg_mesh

