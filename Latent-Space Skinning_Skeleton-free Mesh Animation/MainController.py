import torch
import torch.nn as nn
import numpy as np
from DataSets import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from LSTM_encoder_decoder_attention import Lstm_encoder_decoder_attention
from Lstm_attention_categories import Lstm_encoder_decoder_categories
from mesh_volume_calculator import *
from mean_curvature import *


#training --------------------------------->
def train_evaluate(model,train_loader,evaluate_loader,num_epochs,criterion,optimizer,action,mesh_file, tetrahedrons_file,device):

    simplices = np.load(tetrahedrons_file)
    simplices_torch = torch.tensor(simplices, dtype=torch.long, device=device)
    #print(f"Simplices shape: {simplices.shape}")
    
    faces_np = extract_surface_faces(simplices)
    faces = torch.tensor(faces_np, dtype=torch.long, device=device)

    
    mesh = np.load(mesh_file)
    first_frame = mesh[0]  # Σχήμα (7093, 3)
    first_frame_torch = torch.tensor(first_frame, dtype=torch.float32, device=device) 

    initial_volume = calculate_volume_torch(first_frame_torch, simplices_torch)
    L = build_laplacian_matrix(first_frame.shape[0], faces_np).to(device)
    
    vol_weight = 1.2
    mean_curvature_weight = 0.5
    loss_train_list = []
    loss_eval_list = []
    loss_disPer_train_list = []
    loss_disPer_eval_list = []
    best_val_loss = float('inf')
    patience = 1500
    patience_counter = 0
    for epoch in range(num_epochs+1):
        
        #training ---- >
        model.train()
        epoch_loss,epoch_distPer,epoch_predicted_mesh = model.train_phase(train_loader,criterion,optimizer)

        predicted_mesh_torch = epoch_predicted_mesh[0].view(-1, 3).to(device)
        predicted_volume = calculate_volume_torch(predicted_mesh_torch, simplices_torch)

        volume_loss = (predicted_volume - initial_volume) / (initial_volume)
        curv_loss = curvature_loss(predicted_mesh_torch, first_frame_torch, L)
        # total loss
        total_loss = epoch_loss + (vol_weight * volume_loss)  + (mean_curvature_weight * curv_loss)

        loss_train_list.append(total_loss.item())
        loss_disPer_train_list.append(epoch_distPer)

        if(epoch == num_epochs):
            trainning_output = epoch_predicted_mesh.cpu().detach().numpy()
            
        #evaluation---->
        model.set_evaluation_mode()
        # setted to empty because i dont want the time
        eval_avg_crLoss, eval_DistPer_loss, valuation_mesh  = model.evaluation_phase(evaluate_loader,criterion,"") 

        #-------------------------------------volume------------------------------------------
    
        evaluation_mesh_torch = valuation_mesh[0].view(-1, 3).to(device)
        eval_predicted_volume = calculate_volume_torch(evaluation_mesh_torch, simplices_torch)
        
        eval_volume_loss = (eval_predicted_volume - initial_volume) / (initial_volume )
        eval_curv_loss = curvature_loss(evaluation_mesh_torch, first_frame_torch, L)
        eval_total_loss = eval_avg_crLoss + (vol_weight * eval_volume_loss) + (mean_curvature_weight * eval_curv_loss)

        
        loss_eval_list.append(eval_total_loss.item())
        loss_disPer_eval_list.append(eval_DistPer_loss)

        # when the eval loss is better then we save the model (early stopping)
        if(eval_total_loss<best_val_loss):
            best_val_loss = eval_avg_crLoss
            best_model_state = model.state_dict()  # Save the model state
            last_valuation_output = valuation_mesh.cpu().detach().numpy()
            torch.save(best_model_state, f'best_model_{action}.pth')



        print(f"Epoch [{epoch+1}/{num_epochs}],Loss: {epoch_loss:.4f},DisPer: {epoch_distPer:.4f},Learning Rate: {optimizer.param_groups[0]['lr']:.6f} || ,eval_criterion: {eval_avg_crLoss:.4f},eval_criterion: {best_val_loss:.4f},eval_DistPer: {eval_DistPer_loss:.4f}, || Total Loss: {total_loss.item():.4f}, Total_eval_Loss: {eval_total_loss:.4f}")


    # ploting two plots for the MSE loss and the other for dist per 
    plot_losses(loss_train_list,loss_eval_list,"Total Loss")
    plot_losses(loss_disPer_train_list,loss_disPer_eval_list, "DisPer")
    plot_losses(loss_eval_list,loss_eval_list,"Eval Loss")

    print("this is the training output shape :",trainning_output.shape," this is the evaluation output shape :",last_valuation_output.shape)
    
    makeFile(trainning_output,"training")
    makeFile(last_valuation_output,"evaluation")
    
    return last_valuation_output


def plot_losses(loss_list1, loss_list2, typeOfmatic):
    print("this is the lower value of "+ typeOfmatic +" loss : " + str(min(loss_list1)) )
    plt.figure(figsize=(10, 5))
    
    plt.plot(loss_list1, label=typeOfmatic +" Training ")
    plt.plot(loss_list2, label=typeOfmatic + " Evaluation")
    
    plt.title(f'Losses During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def makeFile(npResults,mode): # this makes a file with the mesh vertices results 
    print("making mesh vertices")
    filepath = f"npy_files/{mode}.npy"
    np.save(filepath,npResults)


def AvgMeshVerticesCalculation(mesh_data,device):
    _mesh_metrices = mesh_data.data.view(mesh_data.data.shape[0],mesh_data.data.shape[-1]//3,3)
    #dim =0 means that the AVG is about the avg of the frames , it will be [27850,3]
    Avg_mesh_metrices = torch.mean(_mesh_metrices, dim=0)
    Avg_mesh_metrices=Avg_mesh_metrices.to(device)
    return Avg_mesh_metrices

def synthesis(model1,model2,model3,evalData1,evalData2,evalData3,device,TypeOfmodel):

    model1.set_evaluation_mode()
    model2.set_evaluation_mode()
    model3.set_evaluation_mode()

    with torch.no_grad():
        
        for _,(bone_matrices1, first_pose_walking, _, _) in enumerate(evalData1):#walking
            _,seq_len,_ = bone_matrices1.size()
            bone_matrices1=bone_matrices1.to(device)
            bone_matrices1 = bone_matrices1[:, :, :624]
            encoder_bone1,(_,_) = model1.encoder(bone_matrices1)          
            
        for _,(bone_matrices2, first_pose_jump, _, _) in enumerate(evalData2):#jump
            _,seq_len,_ = bone_matrices2.size()
            bone_matrices2 = bone_matrices2.to(device)
            bone_matrices2 = bone_matrices2[:, :, :624]
            encoder_bone2,(_,_) = model2.encoder(bone_matrices2)

        for _,(bone_matrices3, _, _, _) in enumerate(evalData3):#turn
            _,seq_len,_ = bone_matrices3.size()
            bone_matrices3 = bone_matrices3.to(device)
            bone_matrices3 = bone_matrices3[:, :, :624]
            encoder_bone3,(_,_) = model3.encoder(bone_matrices3)
        
        if(TypeOfmodel == "first"):
            blended_features_WalkingJump = (encoder_bone1 + encoder_bone2) /2
            decoder_inputs_walkingJump = torch.cat((blended_features_WalkingJump, blended_features_WalkingJump), dim=-1)
            #EDO EVALA KWDIKA GIA NA KOPSO OT TORCH
            min_seq_len = min(encoder_bone1.size(1), encoder_bone3.size(1))
            encoder_bone1 = encoder_bone1[:, :min_seq_len, :]
            encoder_bone3 = encoder_bone3[:, :min_seq_len, :]


            blended_features_WalkingTurn = (encoder_bone1 + encoder_bone3) /2
            decoder_inputs_walkingTurn = torch.cat((blended_features_WalkingTurn, blended_features_WalkingTurn), dim=-1)

            output1 = model2.decoder(decoder_inputs_walkingJump)
            output2 = model1.decoder(decoder_inputs_walkingJump)
            
            output3 = model3.decoder(decoder_inputs_walkingTurn)
            output4 = model1.decoder(decoder_inputs_walkingTurn)

            output1 = output1.squeeze(0)
            output2 = output2.squeeze(0)
            output3 = output3.squeeze(0)
            output4 = output4.squeeze(0)

            print(output1.shape,output2.shape,output3.shape,output4.shape)
            
            #walk and jump
            output1=output1.cpu().detach().numpy()
            output2=output2.cpu().detach().numpy()
            #walk and turn
            output3=output3.cpu().detach().numpy()
            output4=output4.cpu().detach().numpy()


            makeFile(output1,"synthesis1")
            makeFile(output2,"synthesis2")
            makeFile(output3,"synthesis3")
            makeFile(output4,"synthesis4")
        

def CategoriesSynthesis(model,evalTrain,device):
    model.set_evaluation_mode()

    with torch.no_grad():
        for _,(bone_matrices, firstpose , _, _, _) in enumerate(evalTrain):#walking
            
            bone_matrices=bone_matrices.to(device)
            firstpose = firstpose.to(device)
            
            # i feed the model with this vector 
            # you can change this vectors to calculate different synthesis
            category = [0.0,0.0,0.0,1.0,1.0]
            category = torch.tensor(category)
            category = category.to(device)
            category = category.unsqueeze(0)
            # print(category.shape)
            output = model(bone_matrices,firstpose,category)
            output = output.squeeze(0)
            output = output.cpu().detach().numpy()
            
        print(output.shape)
        makeFile(output,"Categories")

def main():    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    mesh_file = "npy_files/meshWalking1.npy"
    tetrahedrons_file = "npy_files/tetrahedrons.npy"

    #----- JUMPS ------>
    JumpBones1 = InputData("npy_files/Jumpbones.npy")
    JumpBones2 = InputData("npy_files/Jumpbones1.npy")
    JumpBones3 = InputData("npy_files/Jumpbones2.npy")
    JumpBones4 = InputData("npy_files/Jumpbones3.npy")
    JumpBones5 = InputData("npy_files/Jumpbones4.npy")
    JumpBonesTest = InputData("npy_files/boneMyJumpTest.npy")
    JumpMesh1 = MeshVerticesDataset("npy_files/JumpMesh.npy")
    JumpMesh2 = MeshVerticesDataset("npy_files/JumpMesh1.npy")
    JumpMesh3 = MeshVerticesDataset("npy_files/JumpMesh2.npy")
    JumpMesh4 = MeshVerticesDataset("npy_files/JumpMesh3.npy")
    JumpMesh5 = MeshVerticesDataset("npy_files/JumpMesh4.npy")
    JumpMeshTest = MeshVerticesDataset("npy_files/meshMyJumpTest.npy")
    JumpAvg1 =AvgMeshVerticesCalculation(JumpMesh1.data,device)
    JumpAvg2 =AvgMeshVerticesCalculation(JumpMesh2.data,device)
    JumpAvg3 =AvgMeshVerticesCalculation(JumpMesh3.data,device)
    JumpAvg4 =AvgMeshVerticesCalculation(JumpMesh4.data,device)
    JumpAvg5 =AvgMeshVerticesCalculation(JumpMesh5.data,device)
    JumpAvgTest = AvgMeshVerticesCalculation(JumpMeshTest.data,device)
    SampleJump1 = Animation(JumpBones1.data.float(),JumpMesh1.data.float(),JumpAvg1)
    SampleJump2 = Animation(JumpBones2.data.float(),JumpMesh2.data.float(),JumpAvg2)
    SampleJump3 = Animation(JumpBones3.data.float(),JumpMesh3.data.float(),JumpAvg3)
    SampleJump4 = Animation(JumpBones4.data.float(),JumpMesh4.data.float(),JumpAvg4)
    SampleJump5 = Animation(JumpBones5.data.float(),JumpMesh5.data.float(),JumpAvg5)
    SampleJumpTest = Animation(JumpBonesTest.data.float(),JumpMeshTest.data.float(),JumpAvgTest)

    #------Rotateshead------->
    rotateBones1 = InputData("npy_files/rotate1bones.npy")
    rotateBones2 = InputData("npy_files/rotate2bones.npy")
    rotateBones3 = InputData("npy_files/rotate3bones.npy")
    rotateBones4 = InputData("npy_files/rotate4bones.npy")
    rotateBones5 = InputData("npy_files/rotate5bones.npy")
    rotateBones6 = InputData("npy_files/rotate6bones.npy")
    rotateMesh1 = MeshVerticesDataset("npy_files/rotate1Mesh.npy")
    rotateMesh2 = MeshVerticesDataset("npy_files/rotate2Mesh.npy")
    rotateMesh3 = MeshVerticesDataset("npy_files/rotate3Mesh.npy")
    rotateMesh4 = MeshVerticesDataset("npy_files/rotate4Mesh.npy")
    rotateMesh5 = MeshVerticesDataset("npy_files/rotate5Mesh.npy")
    rotateMesh6 = MeshVerticesDataset("npy_files/rotate6Mesh.npy")
    rotateAvg1Mesh = AvgMeshVerticesCalculation(rotateMesh1.data,device)
    rotateAvg2Mesh = AvgMeshVerticesCalculation(rotateMesh2.data,device)
    rotateAvg3Mesh = AvgMeshVerticesCalculation(rotateMesh3.data,device)
    rotateAvg4Mesh = AvgMeshVerticesCalculation(rotateMesh4.data,device)
    rotateAvg5Mesh = AvgMeshVerticesCalculation(rotateMesh5.data,device)
    rotateAvg6Mesh = AvgMeshVerticesCalculation(rotateMesh6.data,device)
    SampleRotate1 = Animation(rotateBones1.data.float(), rotateMesh1.data.float(),rotateAvg1Mesh)
    SampleRotate2 = Animation(rotateBones2.data.float(), rotateMesh2.data.float(),rotateAvg2Mesh)
    SampleRotate3 = Animation(rotateBones3.data.float(), rotateMesh3.data.float(),rotateAvg3Mesh)
    SampleRotate4 = Animation(rotateBones4.data.float(), rotateMesh4.data.float(),rotateAvg4Mesh)
    SampleRotate5 = Animation(rotateBones5.data.float(), rotateMesh5.data.float(),rotateAvg5Mesh)
    SampleRotate6 = Animation(rotateBones6.data.float(), rotateMesh6.data.float(),rotateAvg6Mesh)

    #-------Walking ----- >
    walkingBones1 = InputData("npy_files/boneWalking1.npy")
    walkingBones2 = InputData("npy_files/boneWalking2.npy")
    walkingBones3 = InputData("npy_files/boneWalking3.npy")
    walkingBones4 = InputData("npy_files/boneWalking4.npy")
    walkingBones5 = InputData("npy_files/boneWalking5.npy")
    walkingBones6 = InputData("npy_files/boneWalking6.npy")
    walkingBonesTest = InputData("npy_files/boneMyWalkingTest.npy")
    walkingMesh1 = MeshVerticesDataset("npy_files/meshWalking1.npy")
    walkingMesh2 = MeshVerticesDataset("npy_files/meshWalking2.npy")
    walkingMesh3 = MeshVerticesDataset("npy_files/meshWalking3.npy")
    walkingMesh4 = MeshVerticesDataset("npy_files/meshWalking4.npy")
    walkingMesh5 = MeshVerticesDataset("npy_files/meshWalking5.npy")
    walkingMesh6 = MeshVerticesDataset("npy_files/meshWalking6.npy")
    walkingMeshTest = MeshVerticesDataset("npy_files/meshMyWalkingTest.npy")
    WalkingAvgMesh1 = AvgMeshVerticesCalculation(walkingMesh1.data,device)
    WalkingAvgMesh2 = AvgMeshVerticesCalculation(walkingMesh2.data,device)
    WalkingAvgMesh3 = AvgMeshVerticesCalculation(walkingMesh3.data,device)
    WalkingAvgMesh4 = AvgMeshVerticesCalculation(walkingMesh4.data,device)
    WalkingAvgMesh5 = AvgMeshVerticesCalculation(walkingMesh5.data,device)
    WalkingAvgMesh6 = AvgMeshVerticesCalculation(walkingMesh6.data,device)
    walkingAvgMeshTest = AvgMeshVerticesCalculation(walkingMeshTest.data,device)
    SampleWalking1 = Animation(walkingBones1.data.float(), walkingMesh1.data.float(),WalkingAvgMesh1)
    SampleWalking2 = Animation(walkingBones2.data.float(), walkingMesh2.data.float(),WalkingAvgMesh2)
    SampleWalking3 = Animation(walkingBones3.data.float(), walkingMesh3.data.float(),WalkingAvgMesh3)
    SampleWalking4 = Animation(walkingBones4.data.float(), walkingMesh4.data.float(),WalkingAvgMesh4)
    SampleWalking5 = Animation(walkingBones5.data.float(), walkingMesh5.data.float(),WalkingAvgMesh5)
    SampleWalking6 = Animation(walkingBones6.data.float(), walkingMesh6.data.float(),WalkingAvgMesh6)
    SampleWalkingTest = Animation(walkingBonesTest.data.float(), walkingMeshTest.data.float(),walkingAvgMeshTest)


    #----------pointing-------->
    pointingBones1 = InputData("npy_files/bonePointing1.npy")
    pointingBones2 = InputData("npy_files/bonePointing2.npy")
    pointingBones3 = InputData("npy_files/bonePointing3.npy")
    pointingBones4 = InputData("npy_files/bonePointing4.npy")
    pointingMesh1 = MeshVerticesDataset("npy_files/meshPointing1.npy")
    pointingMesh2 = MeshVerticesDataset("npy_files/meshPointing2.npy")
    pointingMesh3 = MeshVerticesDataset("npy_files/meshPointing3.npy")
    pointingMesh4 = MeshVerticesDataset("npy_files/meshPointing4.npy")
    pointingAvgMesh1 = AvgMeshVerticesCalculation(pointingMesh1.data,device)
    pointingAvgMesh2 = AvgMeshVerticesCalculation(pointingMesh2.data,device)
    pointingAvgMesh3 = AvgMeshVerticesCalculation(pointingMesh3.data,device)
    pointingAvgMesh4 = AvgMeshVerticesCalculation(pointingMesh4.data,device)
    SamplePointing1 = Animation(pointingBones1.data.float(),pointingMesh1.data.float(),pointingAvgMesh1)
    SamplePointing2 = Animation(pointingBones2.data.float(),pointingMesh2.data.float(),pointingAvgMesh2)
    SamplePointing3 = Animation(pointingBones3.data.float(),pointingMesh3.data.float(),pointingAvgMesh3)
    SamplePointing4 = Animation(pointingBones4.data.float(),pointingMesh4.data.float(),pointingAvgMesh4)
    #print(SamplePointing1.bones_matrices.shape,"this is the pointing data shape")
    #print(SamplePointing2.bones_matrices.shape,"this is the pointing data shape")
    #print(SamplePointing3.bones_matrices.shape,"this is the pointing data shape")
    #print(SamplePointing4.bones_matrices.shape,"this is the pointing data shape")
    
    #----------Turn------------>
    TurnBones1 = InputData("npy_files/boneTurn1.npy")
    TurnBones2 = InputData("npy_files/boneTurn2.npy")
    TurnBones3 = InputData("npy_files/boneTurn3.npy")
    TurnBones4 = InputData("npy_files/boneTurn4.npy")
    TurnBones5 = InputData("npy_files/boneTurn5.npy")
    TurnBones6 = InputData("npy_files/boneTurn6.npy")
    TurnBones7 = InputData("npy_files/boneTurn7.npy")
    TurnBones8 = InputData("npy_files/boneTurn8.npy")
    TurnBonesTest = InputData("npy_files/boneMyTurnTest.npy")
    TurnMesh1 = MeshVerticesDataset("npy_files/meshTurn1.npy")
    TurnMesh2 = MeshVerticesDataset("npy_files/meshTurn2.npy")
    TurnMesh3 = MeshVerticesDataset("npy_files/meshTurn3.npy")
    TurnMesh4 = MeshVerticesDataset("npy_files/meshTurn4.npy")
    TurnMesh5 = MeshVerticesDataset("npy_files/meshTurn5.npy")
    TurnMesh6 = MeshVerticesDataset("npy_files/meshTurn6.npy")
    TurnMesh7 = MeshVerticesDataset("npy_files/meshTurn7.npy")
    TurnMesh8 = MeshVerticesDataset("npy_files/meshTurn8.npy")
    TurnMeshTest = MeshVerticesDataset("npy_files/meshMyTurnTest.npy")
    TurnAvgMesh1 = AvgMeshVerticesCalculation(TurnMesh1.data,device)
    TurnAvgMesh2 = AvgMeshVerticesCalculation(TurnMesh2.data,device)
    TurnAvgMesh3 = AvgMeshVerticesCalculation(TurnMesh3.data,device)
    TurnAvgMesh4 = AvgMeshVerticesCalculation(TurnMesh4.data,device)
    TurnAvgMesh5 = AvgMeshVerticesCalculation(TurnMesh5.data,device)
    TurnAvgMesh6 = AvgMeshVerticesCalculation(TurnMesh6.data,device)
    TurnAvgMesh7 = AvgMeshVerticesCalculation(TurnMesh7.data,device)
    TurnAvgMesh8 = AvgMeshVerticesCalculation(TurnMesh8.data,device)
    TurnAvgMeshTest = AvgMeshVerticesCalculation(TurnMeshTest.data,device)
    SampleTurn1 = Animation(TurnBones1.data.float(),TurnMesh1.data.float(),TurnAvgMesh1)
    SampleTurn2 = Animation(TurnBones2.data.float(),TurnMesh2.data.float(),TurnAvgMesh2)
    SampleTurn3 = Animation(TurnBones3.data.float(),TurnMesh3.data.float(),TurnAvgMesh3)
    SampleTurn4 = Animation(TurnBones4.data.float(),TurnMesh4.data.float(),TurnAvgMesh4)
    SampleTurn5 = Animation(TurnBones5.data.float(),TurnMesh5.data.float(),TurnAvgMesh5)
    SampleTurn6 = Animation(TurnBones6.data.float(),TurnMesh6.data.float(),TurnAvgMesh6)
    SampleTurn7 = Animation(TurnBones7.data.float(),TurnMesh7.data.float(),TurnAvgMesh7)
    SampleTurn8 = Animation(TurnBones8.data.float(),TurnMesh8.data.float(),TurnAvgMesh8)
    SampleTurnTest = Animation(TurnBonesTest.data.float(),TurnMeshTest.data.float(),TurnAvgMeshTest)

    # with the category vector, encoded in one-hot
    SampleCategoryWalking = AnimationCategory(walkingBones1.data.float(), walkingMesh1.data.float(),WalkingAvgMesh1,[0.0,0.0,0.0,0.0,1.0]) #walk
    SampleCategoryJump = AnimationCategory(JumpBones3.data.float(),JumpMesh3.data.float(),JumpAvg3,[0.0,0.0,0.0,1.0,0.0]) #jump
    SampleCategoryTurn = AnimationCategory(TurnBones2.data.float(),TurnMesh2.data.float(),TurnAvgMesh2,[0.0,0.0,1.0,0.0,0.0]) #turn
    SampleCategoryRotate = AnimationCategory(rotateBones2.data.float(), rotateMesh2.data.float(),rotateAvg2Mesh,[0.0,1.0,0.0,0.0,0.0]) #rotate
    SampleCategoryPoint = AnimationCategory(pointingBones2.data.float(),pointingMesh2.data.float(),pointingAvgMesh2,[1.0,0.0,0.0,0.0,0.0]) #point
    
    
    # Animation of walking
    #[SampleWalking1,SampleWalking2,SampleWalking3,SampleWalking4,SampleWalking5] 
    List_of_samples_walking = [SampleWalking1,SampleWalking2,SampleWalking3,SampleWalking4,SampleWalking5,SampleWalkingTest,SampleJump2,SampleJump4, SampleJump5]
    List_of_evaluation_walking = [SampleWalking4]
    List_for_test_walking = [SampleWalking4] 
    # Animations of jump
    # [SampleJump1,SampleJump2,SampleJump3,SampleJump4,SampleJump5]
    List_of_samples_Jump = [SampleJump1,SampleJump2,SampleJump3,SampleJump4,SampleJump5,SampleJumpTest,SampleWalking2,SampleWalking1,SampleWalking4]
    List_of_evaluation_Jump = [SampleJumpTest]
    List_for_test_Jump = [SampleJump4] 
    # Animation of turning
    List_of_samples_turn = [SampleTurn2,SampleTurn4,SampleTurn6,SampleTurn8,SampleTurnTest,SampleWalking2,SampleWalking1]
    List_of_evaluation_turn = [SampleTurnTest]
    List_for_test_turn = [SampleTurn4]
    # Animation of rotate head
    List_of_samples_Rotate = [SampleRotate2,SampleRotate3,SampleRotate4,SampleRotate5,SampleRotate6,SampleWalking1,SampleWalking2]
    List_of_evaluation_Rotate = [SampleRotate2]
    List_for_test_Rotate = [SampleRotate2] 
    # Animation of pointing
    List_of_samples_Point = [SamplePointing1,SamplePointing2,SampleWalking1,SampleWalking2]
    List_of_evaluation_Point = [SamplePointing2]
    List_for_test_Point = [SamplePointing2] 
    # Animations with categoriers
    List_train_categories = [SampleCategoryWalking,SampleCategoryTurn,SampleCategoryJump,SampleCategoryPoint,SampleCategoryRotate]  
    List_eval_categories = [SampleCategoryJump]

    print("the model will take walking", len(List_of_samples_walking),"Samples")
    print("the other model will take jump",len(List_of_samples_Jump),"Samples")
    print("the other model will take turn",len(List_of_samples_turn),"Samples")
    print("The rotate head animations are" ,len(List_of_samples_Rotate),"Samples")
    print("The Data set for categories of animations has length of", len(List_train_categories))

    #models hyperparameters 
    input_size =  JumpBones1.data.size(1)
    hidden_size = 40
    num_layers = 2
    output_size = JumpMesh1.data.size(1)
    print(input_size,"this is the input size of the model")
    print(output_size,"this is the output size of the model")
    
    #model for walking
    model = Lstm_encoder_decoder_attention(input_size,hidden_size,num_layers,output_size,device).to(device)
    pretrained_path = "best_models/best_model_walk.pth"
    model.load_state_dict(torch.load(pretrained_path))
    model.to(device)
    model.print()
    print(model.count_parameters() ,'this is the parameters of model')
    
    #model for Jump
    modelJump = Lstm_encoder_decoder_attention(input_size,hidden_size,num_layers,output_size,device).to(device)
    pretrained_path = "best_models/best_model_Jump.pth"
    #modelJump.load_state_dict(torch.load(pretrained_path))
    #modelJump.to(device)
    modelJump.print()
    print(modelJump.count_parameters() ,'this is the parameters of model')

    #model for turn
    modelturn = Lstm_encoder_decoder_attention(input_size,hidden_size,num_layers,output_size,device).to(device)
    pretrained_path = "best_models/best_model_turn.pth"
    modelturn.load_state_dict(torch.load(pretrained_path))
    modelturn.to(device)
    modelturn.print()
    print(modelturn.count_parameters(), "this is the parameters of the turn model")

    #model for categories
    category_size = len(List_train_categories)
    modelCategories = Lstm_encoder_decoder_categories(input_size,hidden_size,num_layers,output_size,category_size,device).to(device)
    pretrained_path = "best_models/best_model_categories.pth"
    modelCategories.load_state_dict(torch.load(pretrained_path))
    modelCategories.to(device)
    print(modelCategories)

    #_____________DATASETS WALKING_____________
    batch_size1 = 2
    DataLoaing = CustomDataset(List_of_samples_walking)
    EvalLoading = CustomDataset(List_of_evaluation_walking)
    TestLoad = CustomDataset(List_for_test_walking)    
    train_loader = DataLoader(DataLoaing, batch_size1, shuffle=False)
    evaluate_loader = DataLoader(EvalLoading, batch_size1, shuffle=False) 
    test_loader = DataLoader(TestLoad,batch_size1,shuffle=False)
    #_____________DATASETS JUMP_____________
    batch_size2 = 2
    DataLoaing_Jump = CustomDataset(List_of_samples_Jump)
    EvalLoading_Jump = CustomDataset(List_of_evaluation_Jump)
    TestLoad_Jump = CustomDataset(List_for_test_Jump)    
    train_loader_Jump = DataLoader(DataLoaing_Jump, batch_size2, shuffle=False)
    evaluate_loader_Jump = DataLoader(EvalLoading_Jump, batch_size2, shuffle=False) 
    test_loader_Jump = DataLoader(TestLoad_Jump,batch_size2,shuffle=False)
    #_____________DATASETS TURN_____________
    batch_size3 = 2
    DataLoaing_turn = CustomDataset(List_of_samples_turn)
    EvalLoading_turn = CustomDataset(List_of_evaluation_turn)
    TestLoad_turn = CustomDataset(List_for_test_turn)    
    train_loader_turn = DataLoader(DataLoaing_turn, batch_size3, shuffle=False)
    evaluate_loader_turn = DataLoader(EvalLoading_turn, batch_size3, shuffle=False) 
    test_loader_turn = DataLoader(TestLoad_turn,batch_size3,shuffle=False)

    #________________Rotate Head___________
    batch_size4 = 1
    Data_Loading_rotate = CustomDataset(List_of_samples_Rotate)
    Eval_Loadeing_rotate = CustomDataset(List_of_evaluation_Rotate)
    Test_Loader_rotate = CustomDataset(List_for_test_Rotate)
    train_loader_Rotate = DataLoader(Data_Loading_rotate, batch_size4, shuffle=False)
    evaluate_loader_Rotate = DataLoader(Eval_Loadeing_rotate, batch_size4, shuffle=False)  
    test_loader_Rotate = DataLoader(Test_Loader_rotate,batch_size4,shuffle=False)

    batch_size5 = 1
    Data_Loading_Point = CustomDataset(List_of_samples_Rotate)
    Eval_Loadeing_Point = CustomDataset(List_of_evaluation_Rotate)
    Test_Loader_Point = CustomDataset(List_for_test_Rotate)
    train_loader_Point = DataLoader(Data_Loading_Point, batch_size5, shuffle=False)
    evaluate_loader_Point = DataLoader(Eval_Loadeing_Point, batch_size5, shuffle=False)
    test_loader_Point = DataLoader(Test_Loader_Point,batch_size5,shuffle=False)

    # _____________Dataset for Categorized Animations__________
    batch_size6 = 1 
    DataTrainCategories = CustomDataset(List_train_categories)
    DataEvalCategories = CustomDataset(List_eval_categories)
    Train_loader_categories = DataLoader(DataTrainCategories,batch_size6,shuffle=False)
    Eval_loader_categories = DataLoader(DataEvalCategories,batch_size6,shuffle=False)

    criterion = nn.MSELoss() #based loss function 
    num_epochs = 6500
    
    print("training LSTM_encoder_decoder or with LSTM_encoder_decoder attention  select ---> Train")
    print("Testing the model on test datasets with the choosing of ----> Test")
    print("training a LSTM_encoder_decoder with attention for categories choose ---> categories")
    print("testing motion Synthesis without swapping architecture elements choose --->Synthesis")
    print("test the categorized synthesis choose ---> Evaluate categories")

    user_input = input("Choose how do you want to run this code: ")
    print("you choose ",user_input)
    
    # train 3 identical models for 3 types of animation walk,jump,turn
    if(user_input == "Train"):
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) 
        train_evaluate(model,train_loader,evaluate_loader,num_epochs, criterion, optimizer,"walk", mesh_file, tetrahedrons_file,device)
        
        
        optimizer = torch.optim.Adam(modelJump.parameters(), lr=0.001, weight_decay=1e-5) 
        train_evaluate(modelJump,train_loader_Jump,evaluate_loader_Jump,num_epochs,criterion,optimizer,"Jump",mesh_file, tetrahedrons_file,device)
        
        optimizer = torch.optim.Adam(modelturn.parameters(), lr=0.001, weight_decay=1e-5) 
        train_evaluate(modelturn,train_loader_turn,evaluate_loader_turn,num_epochs,criterion,optimizer,"turn", mesh_file, tetrahedrons_file,device)

    # use this code if you want to train the model with categories 
    elif(user_input == "Categories"):
        optimizer = torch.optim.Adam(modelCategories.parameters(), lr=0.001, weight_decay=1e-5) 
        train_evaluate(modelCategories,Train_loader_categories,Eval_loader_categories,num_epochs, criterion, optimizer,"categories", mesh_file, tetrahedrons_file,device)
    
    #generates outputs with motion blending for categories 
    elif(user_input =="Evaluate categories"):
        model_categories_state_dict = torch.load("best_model_categories.pth")
        modelCategories.load_state_dict(model_categories_state_dict)
        modelCategories.to(device)
        modelCategories.print()
        CategoriesSynthesis(modelCategories,Eval_loader_categories,device)

    elif (user_input=="Test"):

        print("Testing for walk model")
        model_state_dict = torch.load('best_model_walk.pth')
        model.load_state_dict(model_state_dict)
        model.to(device)
        # if the last parameter is allready set to take time, it will generate times for generation of frame and for hole animation
        avg_loss,avg_dist_per_loss,predictions = model.evaluation_phase(test_loader,criterion,"taketime")
        predictions = predictions.cpu().detach().numpy()
        makeFile(predictions,"evaluation_walk")
        print("This is the MSE_loss: ", avg_loss ,"\nThis is the DisPer: ",avg_dist_per_loss,"\n")
        #-------------------------->
        print("Testing for Jump model")
        model_state_dict = torch.load('best_model_Jump.pth')
        modelJump.load_state_dict(model_state_dict)
        modelJump.to(device)
        # if the last parameter is allready set to take time, it will generate times for generation of frame and for hole animation
        avg_loss,avg_dist_per_loss,predictions = modelJump.evaluation_phase(test_loader_Jump,criterion,"taketime")
        predictions = predictions.cpu().detach().numpy()
        makeFile(predictions,"evaluation_jump")
        print("This is the MSE_loss: ", avg_loss ,"\nThis is the DisPer: ",avg_dist_per_loss,"\n")
        #------------------------->
        print("Testing for turn model")
        model_state_dict = torch.load('best_model_turn.pth')
        modelturn.load_state_dict(model_state_dict)
        modelturn.to(device)
        # if the last parameter is allready set to take time, it will generate times for generation of frame and for hole animation
        avg_loss,avg_dist_per_loss,predictions = modelturn.evaluation_phase(test_loader_turn,criterion,"taketime")
        predictions = predictions.cpu().detach().numpy()
        makeFile(predictions,"evaluation_turn")
        print("This is the MSE_loss: ", avg_loss ,"\nThis is the DisPer: ",avg_dist_per_loss,"\n")
        #------------------------->
        print("Testing for rotate model")
        model_state_dict = torch.load('best_model_Rotate.pth')
        modelturn.load_state_dict(model_state_dict)
        modelturn.to(device)
        # if the last parameter is allready set to take time, it will generate times for generation of frame and for hole animation
        avg_loss,avg_dist_per_loss,predictions = modelturn.evaluation_phase(test_loader_Rotate,criterion,"taketime")
        predictions = predictions.cpu().detach().numpy()
        makeFile(predictions,"evaluation_rotate")
        print("This is the MSE_loss: ", avg_loss ,"\nThis is the DisPer: ",avg_dist_per_loss,"\n")
        #------------------------->
        print("Testing for Point model")
        model_state_dict = torch.load('best_model_point.pth')
        modelturn.load_state_dict(model_state_dict)
        modelturn.to(device)
        # if the last parameter is allready set to take time, it will generate times for generation of frame and for hole animation
        avg_loss,avg_dist_per_loss,predictions = modelturn.evaluation_phase(test_loader_Point,criterion,"taketime")
        predictions = predictions.cpu().detach().numpy()
        makeFile(predictions,"evaluation_Point")
        print("This is the MSE_loss: ", avg_loss ,"\nThis is the DisPer: ",avg_dist_per_loss,"\n")
        #--------------------->

    elif (user_input =="Synthesis"):
        # initialize two or more models and trained them to generate the mesh of the animations , 
        #then swap the encoders and visualize the outputs
        #first model to synthesis 
        model_state_dict = torch.load('best_model_walk.pth')
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.print()

        #Second model to synthesis
        model_Jump_state_dict = torch.load("best_model_Jump.pth")
        modelJump.load_state_dict(model_Jump_state_dict)
        modelJump.to(device)
        modelJump.print()

        #third model for extra synthesis
        model_turn_state_dict = torch.load("best_model_turn.pth")
        modelturn.load_state_dict(model_turn_state_dict)
        modelturn.to(device)
        modelturn.print()
        
        TypeOfmodel = "first"
        synthesis(model,modelJump,modelturn,evaluate_loader,evaluate_loader_Jump,evaluate_loader_turn,device,TypeOfmodel)
        

if __name__ == "__main__":
    main()