from __future__ import absolute_import
from __future__ import print_function
import os
import datetime
from shutil import copyfile
from Train_Simulation import Simulation
from Model import TrainModel
from Helpers import Import_Train_Setup, Set_Train_Dir, Sumo_Settings, Save_and_Visualize
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



if __name__ == "__main__":

    config = Import_Train_Setup(configuration_file='Training_Setup.ini')
    CMD = Sumo_Settings(config['gui'], config['sumocfgFileName'], config['maxSteps'])
    Path = Set_Train_Dir(config['modelsPathName'])

    Model = TrainModel(
        config['numLayers'], 
        config['layerWidth'], 
        config['batchSize'], 
        config['learningRate'], 
        config['numStates'], #input_dim
        config['numActions'], #output_dim
        config['maxMemorySize'], 
        config['minMemorySize']
    )
    Simulation = Simulation(
        Model,
        CMD,
        config['gamma'],
        config['maxSteps'],
        config['numCars'],
        config['greenDuration'],
        config['yellowDuration'],
        config['numStates'],
        config['numActions'],
        config['trainingEpochs']
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()
    
    while episode < config['numEpisodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['numEpisodes']))
        epsilon = 1.0 - (episode / config['numEpisodes'])  # set the epsilon for this episode according  to epsilon-greedy policy
        simulation_time, training_time = Simulation.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", Path)

    Model.save_model(Path)

    copyfile(src='Training_Setup.ini', dst=os.path.join(Path, 'Training_Setup.ini'))

    Save_and_Visualize(path=Path,dpi=96,data=Simulation.reward_store, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')
    Save_and_Visualize(path=Path,dpi=96,data=Simulation.cumulative_wait_store, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')
    Save_and_Visualize(path=Path,dpi=96,data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')