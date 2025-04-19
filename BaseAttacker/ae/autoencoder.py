import os
import sys
from collections import namedtuple
from os.path import abspath, dirname

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if "../" not in sys.path:
    sys.path.append("../")
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.utils_buf import Memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' import configuration '''
from yacs.config import CfgNode as CN

yaml_name = os.path.join(dirname(dirname(abspath(__file__))), "config", "config_default.yaml")
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

LEN_TRAJECTORY = config.AE.LEN_TRAJECTORY
SEQ_LEN = config.AE.SEQ_LEN
EMBEDDING_SIZE = config.AE.EMBEDDING_SIZE
MEMORY_SIZE = config.AE.MEMORY_SIZE


""" LSTM-AutoEncoder Memory """
Transition = namedtuple('Transition', ('pre_state', 'pre_action', 'state', 'action'))


""" Define the Encoder(LSTM) Network"""

class Encoder_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer):
        super(Encoder_LSTM, self).__init__()

        self.num_layer = num_layer
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layer)

    def forward(self, x):
        # Create hidden states based on input dimensions
        h0 = torch.zeros(self.num_layer, x.size(1), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layer, x.size(1), self.hidden_size).to(device)

        # Forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Return last hidden state
        embedding = hn[-1, :, :]
        return embedding.data


""" Define the Decoder(FC) Network"""

class Decoder_FC(nn.Module):
    def __init__(self, input_size, output_size, fc1_units=36, fc2_units=36):
        super(Decoder_FC, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.ln1 = nn.LayerNorm(fc1_units)
        self.fc2 = nn.Linear(fc1_units, output_size)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        x = F.softmax(x, dim=2)
        return x


""" Define the Decoder(LSTM) Network """

class Decoder_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size: int, num_layer, output_size):
        super(Decoder_LSTM, self).__init__()

        self.num_layer = num_layer
        self.hidden_size = int(hidden_size)

        self.lstm = nn.LSTM(input_size, int(hidden_size), num_layer)
        self.fc = nn.Linear(int(hidden_size), output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layer, x.size(1), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layer, x.size(1), self.hidden_size).to(device)

        out, (hn, cn) = self.lstm(x, (h0, c0))

        out = self.fc(out[:, :, :])
        out = F.softmax(out, dim=2)
        return out


""" Construct Encoder + Decoder """

class Encoder_Decoder(nn.Module):
    def __init__(self, encoder, decoder_fc, decoder_lstm):
        super(Encoder_Decoder, self).__init__()
        self.encoder = encoder
        self.decoder_fc = decoder_fc
        self.decoder_lstm = decoder_lstm

    def forward(self, enc_in, dec_fc_in, dec_lstm_in):
        # encoder
        embedding = self.encoder(enc_in)

        # decoder-fc
        z = embedding.unsqueeze(1)
        z = torch.cat(enc_in.size(0)*[z],1)
        x1 = torch.cat((dec_fc_in, z), 2)
        y1 = self.decoder_fc(x1)

        # decoder-lstm
        z = embedding.unsqueeze(0)
        z = torch.cat(enc_in.size(0)*[z],0)
        x2 = torch.cat((dec_lstm_in, z), 2)
        y2 = self.decoder_lstm(x2)

        return y1, y2


class EnvAutoEncoder():

    def __init__(self, env, enc_in_size, enc_out_size, enc_num_layer,
                 dec_fc_in_size, dec_fc_out_size, dec_lstm_in_size, dec_lstm_out_size, dec_lstm_num_layer,
                 seq_len, embedding_len, n_epochs=50, lr=0.001):

        self.n_epochs = n_epochs
        self.lr = lr

        self.Enc_Model = Encoder_LSTM(enc_in_size, enc_out_size, enc_num_layer).to(device)
        self.Dec_FC_Model = Decoder_FC(dec_fc_in_size, dec_fc_out_size).to(device)
        self.Dec_LSTM_Model = Decoder_LSTM(dec_lstm_in_size, dec_lstm_out_size, dec_lstm_num_layer, dec_lstm_out_size).to(device)

        self.Model = Encoder_Decoder(self.Enc_Model, self.Dec_FC_Model, self.Dec_LSTM_Model).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.Model.parameters(), self.lr)

        self.enc_in_size = enc_in_size
        self.enc_out_size = enc_out_size
        self.enc_num_layer = enc_num_layer

    def _train(self, memory_A, memory_B):

        self.Model.train()

        n_trajectory = MEMORY_SIZE//SEQ_LEN

        for epoch in range(self.n_epochs):

            train_loss = 0.0  # monitor training loss

            for n in range(n_trajectory-1):
                i_start = n*SEQ_LEN
                i_end = i_start + SEQ_LEN

                '''  Input to Encoder  '''
                # Actual(A) Set
                A_transitions_enc = memory_A.memory[i_start: i_end]
                A_batch_enc = Transition(*zip(*A_transitions_enc))

                A_state_enc = torch.FloatTensor([A_batch_enc.state]).view(-1,1).to(device)
                A_action_enc = torch.FloatTensor([A_batch_enc.action]).view(-1,1).to(device)

                A_enc_IN = torch.cat((A_state_enc, A_action_enc),1).unsqueeze(1)

                # Desired (B) Set
                B_transitions_enc = memory_B.memory[i_start: i_end]
                B_batch_enc = Transition(*zip(*B_transitions_enc))

                B_state_enc = torch.FloatTensor([B_batch_enc.state]).view(-1,1).to(device)
                B_action_enc = torch.FloatTensor([B_batch_enc.action]).view(-1,1).to(device)

                B_enc_IN = torch.cat((B_state_enc, B_action_enc),1).unsqueeze(1)

                # Input to Encoder
                enc_IN = torch.cat((A_enc_IN, B_enc_IN), 1)

                '''  Input to Decoder  '''
                # Actual (A) Set
                A_transition_dec = memory_A.memory[i_start+SEQ_LEN : i_end+SEQ_LEN]
                A_batch_dec = Transition(*zip(*A_transition_dec))

                A_pre_state_dec = torch.FloatTensor([A_batch_dec.pre_state]).view(-1,1).to(device)
                A_pre_action_dec = torch.FloatTensor([A_batch_dec.pre_action]).view(-1,1).to(device)
                A_state_dec = torch.FloatTensor([A_batch_dec.state]).view(-1,1).to(device)
                A_action_dec = torch.FloatTensor([A_batch_dec.action]).view(-1,1).to(device)

                A_dec_fc_IN = A_state_dec
                A_dec_fc_TARGET = A_action_dec.long().view(1, SEQ_LEN)

                A_dec_lstm_IN = torch.cat((A_pre_state_dec, A_pre_action_dec),1)
                A_dec_lstm_TARGET = A_state_dec.long().view(1, SEQ_LEN)

                # Desired (B) Set
                B_transition_dec = memory_B.memory[i_start+SEQ_LEN : i_end+SEQ_LEN]
                B_batch_dec = Transition(*zip(*B_transition_dec))

                B_pre_state_dec = torch.FloatTensor([B_batch_dec.pre_state]).view(-1,1).to(device)
                B_pre_action_dec = torch.FloatTensor([B_batch_dec.pre_action]).view(-1,1).to(device)
                B_state_dec = torch.FloatTensor([B_batch_dec.state]).view(-1,1).to(device)
                B_action_dec = torch.FloatTensor([B_batch_dec.action]).view(-1,1).to(device)

                B_dec_fc_IN = B_state_dec
                B_dec_fc_TARGET = B_action_dec.long().view(1, SEQ_LEN)

                B_dec_lstm_IN = torch.cat((B_pre_state_dec, B_pre_action_dec),1)
                B_dec_lstm_TARGET = B_state_dec.long().view(1, SEQ_LEN)

                # Input to Dec_FC & Dec_LSTM
                dec_fc_IN = torch.cat((A_dec_fc_IN.unsqueeze(0), B_dec_fc_IN.unsqueeze(0)), 0)
                dec_lstm_IN = torch.cat((A_dec_lstm_IN.unsqueeze(1), B_dec_lstm_IN.unsqueeze(1)), 1)

                dec_fc_TARGET = torch.cat((A_dec_fc_TARGET, B_dec_fc_TARGET), 0).view(-1)
                dec_lstm_TARGET = torch.cat((A_dec_lstm_TARGET, B_dec_lstm_TARGET),0).view(-1)


                ''' Optimize '''
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()

                # forward pass: compute predicted outputs by passing inputs to the model
                out1, out2 = self.Model(enc_IN, dec_fc_IN, dec_lstm_IN)
                # reshape output & target
                out1 = torch.cat((out1[0,:,:], out1[1,:,:]), 0)
                out2 = torch.cat((out2[:,0,:], out2[:,1,:]), 0)

                # calculate the loss
                loss = self.criterion(out1, dec_fc_TARGET) + self.criterion(out2, dec_lstm_TARGET)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # perform a single optimization step (parameter update)
                self.optimizer.step()

                # update running training loss
                train_loss += loss.item()

            total_size = (n_trajectory-1)*SEQ_LEN*2

            train_loss = train_loss/total_size
#             print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
#             print("Train Encoder-Decoder Neural Network ... ")


    def _eval(self, memory_A, memory_B):

        test_loss = 0.0
        correct_stat_1 = 0
        correct_stat_2 = 0

        self.Model.eval() # prep model for *evaluation*

        n_trajectory = MEMORY_SIZE//SEQ_LEN

        # for data, target in test_loader:
        for n in range(n_trajectory-1):

            i_start = n*SEQ_LEN
            i_end = i_start + SEQ_LEN

            '''  Input to Encoder  '''
            # Actual(A) Set
            A_transitions_enc = memory_A.memory[i_start: i_end]
            A_batch_enc = Transition(*zip(*A_transitions_enc))

            A_state_enc = torch.FloatTensor([A_batch_enc.state]).view(-1,1).to(device)
            A_action_enc = torch.FloatTensor([A_batch_enc.action]).view(-1,1).to(device)

            A_enc_IN = torch.cat((A_state_enc, A_action_enc),1).unsqueeze(1)

            # Desired (B) Set
            B_transitions_enc = memory_B.memory[i_start: i_end]
            B_batch_enc = Transition(*zip(*B_transitions_enc))

            B_state_enc = torch.FloatTensor([B_batch_enc.state]).view(-1,1).to(device)
            B_action_enc = torch.FloatTensor([B_batch_enc.action]).view(-1,1).to(device)

            B_enc_IN = torch.cat((B_state_enc, B_action_enc),1).unsqueeze(1)

            # Input to Encoder
            enc_IN = torch.cat((A_enc_IN, B_enc_IN), 1)

            '''  Input to Decoder  '''
            # Actual (A) Set
            A_transition_dec = memory_A.memory[i_start+SEQ_LEN : i_end+SEQ_LEN]
            A_batch_dec = Transition(*zip(*A_transition_dec))

            A_pre_state_dec = torch.FloatTensor([A_batch_dec.pre_state]).view(-1,1).to(device)
            A_pre_action_dec = torch.FloatTensor([A_batch_dec.pre_action]).view(-1,1).to(device)
            A_state_dec = torch.FloatTensor([A_batch_dec.state]).view(-1,1).to(device)
            A_action_dec = torch.FloatTensor([A_batch_dec.action]).view(-1,1).to(device)

            A_dec_fc_IN = A_state_dec
            A_dec_fc_TARGET = A_action_dec.long().view(1, SEQ_LEN)

            A_dec_lstm_IN = torch.cat((A_pre_state_dec, A_pre_action_dec),1)
            A_dec_lstm_TARGET = A_state_dec.long().view(1, SEQ_LEN)

            # Desired (B) Set
            B_transition_dec = memory_B.memory[i_start+SEQ_LEN : i_end+SEQ_LEN]
            B_batch_dec = Transition(*zip(*B_transition_dec))

            B_pre_state_dec = torch.FloatTensor([B_batch_dec.pre_state]).view(-1,1).to(device)
            B_pre_action_dec = torch.FloatTensor([B_batch_dec.pre_action]).view(-1,1).to(device)
            B_state_dec = torch.FloatTensor([B_batch_dec.state]).view(-1,1).to(device)
            B_action_dec = torch.FloatTensor([B_batch_dec.action]).view(-1,1).to(device)

            B_dec_fc_IN = B_state_dec
            B_dec_fc_TARGET = B_action_dec.long().view(1, SEQ_LEN)

            B_dec_lstm_IN = torch.cat((B_pre_state_dec, B_pre_action_dec),1)
            B_dec_lstm_TARGET = B_state_dec.long().view(1, SEQ_LEN)

            # Input to Dec_FC & Dec_LSTM
            dec_fc_IN = torch.cat((A_dec_fc_IN.unsqueeze(0), B_dec_fc_IN.unsqueeze(0)), 0)
            dec_lstm_IN = torch.cat((A_dec_lstm_IN.unsqueeze(1), B_dec_lstm_IN.unsqueeze(1)), 1)

            dec_fc_TARGET = torch.cat((A_dec_fc_TARGET, B_dec_fc_TARGET), 0).view(-1)
            dec_lstm_TARGET = torch.cat((A_dec_lstm_TARGET, B_dec_lstm_TARGET),0).view(-1)


            """ output & loss """
            out_1, out_2 = self.Model(enc_IN, dec_fc_IN, dec_lstm_IN)
            # reshape output & target
            out_1 = torch.cat((out_1[0,:,:], out_1[1,:,:]), 0)
            out_2 = torch.cat((out_2[:,0,:], out_2[:,1,:]), 0)

            # calculate the loss
            loss = self.criterion(out_1, dec_fc_TARGET) + self.criterion(out_2, dec_lstm_TARGET)
            # update test loss
            test_loss += loss.item()


            """ correctness of \\pi(a|s) """
            # convert output probabilities to predicted class
            _, pred = torch.max(out_1, 1)

            # compare predictions to true label
            correct_1 = np.squeeze(pred.eq(dec_fc_TARGET.data.view_as(pred)))

            for i in range(len(correct_1)):
                if correct_1[i].item() == True:
                    correct_stat_1 += 1


            """ correctness of T(s'|s,a) """
            # convert output probabilities to predicted class
            _, pred = torch.max(out_2, 1)

            # compare predictions to true label
            correct_2 = np.squeeze(pred.eq(dec_lstm_TARGET.data.view_as(pred)))

            for i in range(len(correct_2)):
                if correct_2[i].item() == True:
                    correct_stat_2 += 1


        # calculate and print avg test loss
        total_size = (n_trajectory-1)*SEQ_LEN*2

        test_loss = test_loss/total_size
    #     print('Test Loss: {:.6f}\n'.format(test_loss))
        accuracy_rate_1 = (correct_stat_1/total_size)*100
    #     print('Accuracy Rate of \PI: {:.2f}%\n'.format(accuracy_rate_1))
        accuracy_rate_2 = (correct_stat_2/total_size)*100
    #     print('Accuracy Rate of T: {:.2f}%\n'.format(accuracy_rate_2))
        print(f'Embedding Accuracy: PI={accuracy_rate_1:.2f}%, T={accuracy_rate_2:.2f}%')



    def _embedding_BATCH(self, memory_A, memory_B):
        self.Model.eval() # prep model for *evaluation*

        n_trajectory = MEMORY_SIZE//SEQ_LEN

        zA = []
        zB = []

        # Convert lists to Memory objects if needed
        if isinstance(memory_A, list):
            mem_A = Memory(MEMORY_SIZE)
            for item in memory_A:
                if isinstance(item, tuple):
                    mem_A.push(*item)
                else:
                    mem_A.push(item[0], item[1])  # Assuming each item is [state, action]
            memory_A = mem_A

        if isinstance(memory_B, list):
            mem_B = Memory(MEMORY_SIZE)
            for item in memory_B:
                if isinstance(item, tuple):
                    mem_B.push(*item)
                else:
                    mem_B.push(item[0], item[1])  # Assuming each item is [state, action]
            memory_B = mem_B

        for n in range(n_trajectory):
            # Input Data
            i_start = n*SEQ_LEN
            i_end = i_start + SEQ_LEN

            '''  Input to Encoder  '''
            # Actual(A) Set
            A_transitions = memory_A.memory[i_start: i_end]
            A_batch = Transition(*zip(*A_transitions))

            A_state = torch.FloatTensor([A_batch.state]).view(-1,1).to(device)
            A_action = torch.FloatTensor([A_batch.action]).view(-1,1).to(device)

            A_IN = torch.cat((A_state, A_action),1).unsqueeze(1)

            # Desired (B) Set
            B_transitions = memory_B.memory[i_start: i_end]
            B_batch = Transition(*zip(*B_transitions))

            B_state = torch.FloatTensor([B_batch.state]).view(-1,1).to(device)
            B_action = torch.FloatTensor([B_batch.action]).view(-1,1).to(device)

            B_IN = torch.cat((B_state, B_action),1).unsqueeze(1)

            # Input to Encoder
            IN = torch.cat((A_IN, B_IN), 1)

            ''' Get embedding '''
            z = self.Enc_Model(IN)

            zA.append(z[0].cpu().data.numpy())
            zB.append(z[1].cpu().data.numpy())

        zA = np.vstack(zA)
        zB = np.vstack(zB)

        return zA, zB


    def _save(self, filename):
        torch.save(self.Model.state_dict(), filename + "_AutoEncoder")

    def _load(self, filename):
        load_model = self.Model.load_state_dict(torch.load(filename))

        return load_model

    def encode_environment(self, altitude):
        """Encode environment state (altitude) into embedding."""
        # Reshape altitude to match LSTM input expectations: (seq_len, batch, input_size)
        env_tensor = torch.from_numpy(altitude).float()
        env_tensor = env_tensor.view(-1, 1, self.enc_in_size)  # (seq_len, batch=1, features)

        # Forward pass through encoder LSTM
        embedding = self.Enc_Model(env_tensor)

        return embedding

    def combine_embeddings(self, policy_embedding, env_embedding):
        """Combine policy and environment embeddings into single state representation."""
        return torch.cat((policy_embedding, env_embedding), dim=1)



if __name__ == "__main__":

    # env
    from envs.env3D_4x4 import Grid3D
    env = Grid3D()

    """ Intialize AutoEncoder """
    ae_enc_IN = 2
    ae_enc_OUT = EMBEDDING_SIZE # embedding size
    ae_enc_N_layer = 1

    ae_dec_fc_IN = ae_enc_OUT + 1
    ae_dec_fc_OUT = env.nA

    ae_dec_lstm_IN = ae_enc_OUT + 2
    ae_dec_lstm_OUT = env.nS
    ae_dec_lstm_N_layer = 1

    ae_args = {
        "env": env,
        "enc_in_size": ae_enc_IN,
        "enc_out_size": ae_enc_OUT,
        "enc_num_layer": ae_enc_N_layer,
        "dec_fc_in_size": ae_dec_fc_IN,
        "dec_fc_out_size": ae_dec_fc_OUT,
        "dec_lstm_in_size": ae_dec_lstm_IN,
        "dec_lstm_out_size": ae_dec_lstm_OUT,
        "dec_lstm_num_layer": ae_dec_lstm_N_layer,
        "seq_len": SEQ_LEN,
        "embedding_len": EMBEDDING_SIZE,
        "n_epochs": 2,
        "lr": 0.001,
    }

    ae_Model = EnvAutoEncoder(**ae_args)


