import time
import torch
import torch.nn as nn
import torch.nn.functional as F
  

class Encoder_BoneMovement(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,device):
        super(Encoder_BoneMovement, self).__init__()
        self.device= device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
     
    def forward(self, input_data):
        self.lstm.flatten_parameters()
        hidden = self.init_hidden(input_data.size(0))
        output_encoder, (hidden_state, cell_state) = self.lstm(input_data,hidden)
        output_encoder = nn.Tanh()(output_encoder)
        return output_encoder,(hidden_state, cell_state)
    
    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))
        return hidden

class Encoder_firstPose(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,device):
        super(Encoder_firstPose, self).__init__()
        self.device= device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size ,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)

    def forward(self, input_data):
        x = F.tanh(self.fc1(input_data))
        x = F.tanh(self.fc2(x))
        return x 
  
class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(feature_dim*2, feature_dim*2),
            nn.Tanh(),
            nn.Linear(feature_dim*2, 1)
        )

    def forward(self, encoder_output, encoder_first_pose):
        combined_features = torch.cat((encoder_output, encoder_first_pose), dim=-1)
        attention_weights = self.attention_weights_layer(combined_features)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        attended_features = torch.sum(attention_weights * combined_features, dim=1)
        return attended_features

class Decoder(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size):
        super(Decoder, self).__init__()
        
        self.lstm = nn.LSTM(hidden_size*2, hidden_size*2, num_layers, batch_first=True)
        self.input_decoder = nn.Linear(hidden_size*2,hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, decoder_input):
        self.lstm.flatten_parameters()
        output, _ = self.lstm(decoder_input)
        output = self.input_decoder(output)
        output = self.layer_norm(output)
        output = self.fc(output)        
        return output
        
class Lstm_encoder_decoder_attention(nn.Module):
  
    def __init__(self,input_size,hidden_size,num_layers,output_size,device):
        super(Lstm_encoder_decoder_attention,self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.attention = Attention(hidden_size)
        self.encoder = Encoder_BoneMovement(input_size,hidden_size,num_layers,device)
        self.encoder_firstPose = Encoder_firstPose(output_size,hidden_size,num_layers,device)
        self.decoder =Decoder(hidden_size,num_layers,output_size)
          
    def forward(self,input_bone,first_frame_pose): #rest pose is like [frames,vertices] = [43,83550] input_bones = [43,624]       
        
        batch_size,seq_len,_ = input_bone.size()
        self.batch_size = batch_size
        #edo evala code ego
        input_bone = input_bone[:, :, :624]
        # print(input_bone.shape,"bones input")
        # print(first_frame_pose.shape,"first_frame_pose")
        encoder_output,(hidden_bones,cell_state_bones) = self.encoder(input_bone)
        encoder_first_pose = self.encoder_firstPose(first_frame_pose)
       
        encoder_first_pose = encoder_first_pose.unsqueeze(1).expand(-1,seq_len,-1)
        # print(encoder_output.shape,encoder_first_pose.shape , "encoders outputs --- > attention _ inputs ")
        decoder_input = self.attention(encoder_output,encoder_first_pose)
        # print(decoder_input.shape,"decoder_input,attentionoutput")
        decoder_input = decoder_input.unsqueeze(1).expand(-1,seq_len,-1)
        output = self.decoder(decoder_input)
        # print(output.shape,"output")

        return output
    
    def set_evaluation_mode(self):
        self.eval()
        self.encoder.eval()
        self.encoder_firstPose.eval()
        self.attention.eval()
        self.decoder.eval()

    def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print(self):
         print(self)

    #------------------------Dist Per function ------------------------------->
    def DistPer(self,A_orig,A_approx,Avg_mesh_metrices): 
        
        Avg_mesh_metrices = Avg_mesh_metrices.to(self.device)
        Avg_mesh_metrices = Avg_mesh_metrices[0,:]
        # print(Avg_mesh_metrices.shape)

        #reshape because of they are vertices and we must analyze it as vertices (frames,mesh_values,xyz)
        A_orig = A_orig.view(A_orig.shape[0],A_orig.shape[-1]//3,3)   
        A_approx = A_approx.view(A_approx.shape[0],A_approx.shape[-1]//3,3)

        #calculations
        diff_Aorig_Aapprox = torch.norm(A_orig - A_approx, p="fro")
        diff_Aorig_Aavg = torch.norm(A_orig - Avg_mesh_metrices, p="fro")
        disPervalue =  100*(diff_Aorig_Aapprox / (diff_Aorig_Aavg))

        return disPervalue
    
    
    
    def train_phase(self, train_loader, criterion ,optimizer):
        
        total_loss = 0.0
        total_DistPer_loss = 0.0
        for batch_idx,(bone_matrices, first_frame, target_mesh, mesh_avg) in enumerate(train_loader):
            bone_matrices = bone_matrices.to(self.device)
            first_frame = first_frame.to(self.device)
            target_mesh =  target_mesh.to(self.device)
            mesh_avg = mesh_avg.to(self.device)
            
            predicted_mesh =  self.forward(bone_matrices,first_frame)
            predicted_mesh = predicted_mesh.squeeze(0)
            target_mesh = target_mesh.squeeze(0)

            loss = criterion(predicted_mesh,target_mesh)
            if(self.batch_size > 1 ):
                # print(self.batch_size)
                distPer = self.DistPer(target_mesh[self.batch_size-1],predicted_mesh[self.batch_size-1],mesh_avg)
            else:
                distPer = self.DistPer(target_mesh,predicted_mesh,mesh_avg)

            #backpropagation 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_DistPer_loss += distPer.item()            
            
        epoch_loss_disper = total_DistPer_loss / len(train_loader)        
        epoch_loss = total_loss / len(train_loader)  
        return epoch_loss , epoch_loss_disper , predicted_mesh 

 

    def evaluation_phase(self,evaluate_loader,criterion,mode):
        self.set_evaluation_mode()
        total_loss = 0.0
        total_loss_distPer = 0.0
        start_time = 0 
        end_time = 0

        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(f"Warning: Parameter {name} is frozen during evaluation.")

        with torch.no_grad():
            for batch_idx,(bone_matrices, first_frame, target_valuation_mesh, mesh_avg) in enumerate(evaluate_loader):

                bone_matrices = bone_matrices.to(self.device)
                first_frame = first_frame.to(self.device)
                target_valuation_mesh = target_valuation_mesh.to(self.device)
                mesh_avg = mesh_avg.to(self.device)
                seq_len = bone_matrices.shape[1]
                
                
                if (mode=="taketime"):
                    start_time = time.time()

                predicted_valuation_mesh = self.forward(bone_matrices,first_frame)

                if (mode=="taketime"):
                    end_time = time.time()
                    elapse_time_animation = end_time - start_time
                    elapse_time_frame = elapse_time_animation/seq_len
                    print(f"Time for Mesh generation for hole animation: {elapse_time_animation} seconds")
                    print(f"Time for Mesh generation per frame: {elapse_time_frame} seconds")

                predicted_valuation_mesh = predicted_valuation_mesh.squeeze(0)
                target_valuation_mesh = target_valuation_mesh.squeeze(0)

                loss = criterion(predicted_valuation_mesh,target_valuation_mesh)
                distPer = self.DistPer(target_valuation_mesh,predicted_valuation_mesh,mesh_avg)

                total_loss += loss.item()
                total_loss_distPer += distPer.item()

            avg_loss = total_loss / len(evaluate_loader)
            avg_dist_per_loss =total_loss_distPer / len(evaluate_loader)

            return avg_loss , avg_dist_per_loss , predicted_valuation_mesh