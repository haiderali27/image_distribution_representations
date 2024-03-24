import os 
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from scipy.stats import wasserstein_distance
import random
from skimage.color import rgb2gray

import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy import arange
import time
import concurrent.futures

import pickle





class CalculateImageDistribution:

    dict_datasets={}
    dict_matrices={}
    dict_colors={}
    dict_dataset_description = {}
    mvtech_train_len = -1 

    def save_dicts(self, file_name):
        with open(file_name+'.pkl', 'wb') as file:
            pickle.dump(self.dict_matrices, file)
            pickle.dump(self.dict_colors, file)
            pickle.dump(self.dict_dataset_description, file)




    def load_dicts(self, file_name):
        with open(file_name, 'rb') as file:
            self.dict_matrices = pickle.load(file)
            self.dict_colors = pickle.load(file)
            self.dict_dataset_description = pickle.load(file)


        
    def __init__(self):
        dict_datasets={}

    def set_mvtech_dataset(self, image_path, dataset_name, resizeImage=True, resizeH=300, resizeW=300, load_all=False, divisions=1, datasetSize=50, include_grey=False):
        self.set_dataset(image_path+'/train/good', dataset_name+"_train",  'k', resizeImage, resizeH, resizeW, load_all, divisions, datasetSize, include_grey)
        self.mvtech_train_len = len(self.dict_datasets[dataset_name+"_train"])

        base_path = image_path +'/test/'
        sub_dirs = os.listdir(base_path)
        for subdir in sub_dirs:
            color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            self.set_dataset(os.path.join( base_path, subdir), dataset_name+"_test_"+subdir, color, resizeImage, resizeH, resizeW, load_all, divisions, datasetSize, include_grey)
    
    def merge_dataset(self, dataA:str, dataB:str):
         self.dict_datasets[dataA+dataB] = self.dict_datasets[dataA]+self.dict_datasets[dataB]
         color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
         self.dict_colors[dataA+dataB] = color   
         self.dict_dataset_description[dataA+dataB] = f'dataset_name:{dataA+dataB}, color:{color}, , loaded_data_len:{len(self.dict_datasets[dataA+dataB])}'  

    def merge_datasets(self, datasets:[str]):
         merged_ds =""
         data=[]
         for dataset in datasets:
              merged_ds+=dataset
              data += self.dict_datasets[dataset]
         self.dict_datasets[merged_ds] = data
         color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
         self.dict_colors[merged_ds] = color   
         self.dict_dataset_description[merged_ds] = f'dataset_name:{merged_ds}, color:{color}, , loaded_data_len:{len(self.dict_datasets[merged_ds])}'  


    def set_dataset(self, image_path, dataset_name, color,resizeImage=True, resizeH=300, resizeW=300, load_all=False, divisions=1, datasetSize=50, include_grey=False):    
        self.dict_datasets[dataset_name] = self.load_images(image_path, resizeImage, resizeH, resizeW, load_all, divisions, datasetSize, include_grey) 
        self.dict_colors[dataset_name] = color   
        self.dict_dataset_description[dataset_name] = f'dataset_name:{dataset_name}, color:{color}, resizeH:{resizeH}, resizeW:{resizeW}, load_all:{load_all}, divisions:{divisions}, datasetSize:{datasetSize}, loaded_data_len:{len(self.dict_datasets[dataset_name])}'  
    
    def load_images(self, imagePath, resizeImage=True, resizeH=300, reSizeW=300, load_all=False,  divisions=1, dataSize=50, include_grey=False):
        data = []; 
        if(len(os.listdir(imagePath))<=dataSize):
              load_all=True
        if load_all:
            for img in os.listdir(imagePath):
                    im = Image.open(imagePath +'/'+ img)
                    if resizeImage:
                        im = im.resize((resizeH,reSizeW))

                    if len(im.getbands())==1:    
                          if include_grey is True:
                            im = np.array(im)
                            data.append(im)
                    else: 
                         im = np.array(im)
                         data.append(im)
        else:    
            number_of_images = len(os.listdir(imagePath))
            #print(number_of_images, '############', imagePath, os.listdir(imagePath) )
            if(divisions==1):
                
                data_start = 0 if dataSize > number_of_images  else random.randint(0, number_of_images-dataSize)
                data_end = number_of_images if dataSize > number_of_images else  data_start+dataSize
                #print(f'dataStart: {data_start}, dataEnd:{data_end}')
                i = 0
                for img in os.listdir(imagePath):
                    if(i<data_start):
                        i+=1
                        continue;
                    if (i>=data_end):
                        break;
                    im = Image.open(imagePath +'/'+ img)
                    if resizeImage:
                        im = im.resize((resizeH,reSizeW))


                    if len(im.getbands())==1:    
                          if include_grey is True:
                            im = np.array(im)
                            data.append(im)
                    else: 
                         im = np.array(im)
                         data.append(im)    
                    #im = np.array(im)
                    #data.append(im)
                    i+=1
            else:
                data_divisions = number_of_images/divisions
                data_divisions = int(data_divisions)
                dataSizes = dataSize/divisions
                dataSizes = int(dataSizes)
                for chunk in range(divisions):  
                    data_start = random.randint((chunk*data_divisions), ((chunk+1)*data_divisions)-dataSizes)
                    data_end = data_start+dataSizes
                    #print(f'dataSizes:{dataSize}, data_start:{data_start}, data_end:{data_end}')
                    j = 0
                    for img in os.listdir(imagePath):
                        if(j<data_start):
                            j+=1
                            continue;
                        if (j>=data_end):

                            break;
                        im = Image.open(imagePath +'/'+ img)
                        if resizeImage:
                            im = im.resize((resizeH,reSizeW))
                        


                        if len(im.getbands())==1:    
                          if include_grey is True:
                            im = np.array(im)
                            data.append(im)
                        else: 
                         im = np.array(im)
                         data.append(im)
                        #im = np.array(im)
                        #data.append(im)
                        j+=1

        #print(f'len_data:{len(data)}')
        return data
    
    def calculate_all_distributions(self, matrix_name='ssim', is_rgb=True, channels = 3, winsize=5):
         for key in self.dict_datasets.keys():
              self.calculate_distribution_dataset(key, matrix_name, is_rgb=is_rgb, channels=channels, winsize=winsize)
         
    
    def calculate_distribution_dataset(self, dataset_name, martix_name='ssim', is_rgb=True, channels = 3, winsize=5):
         matrix = []
         if martix_name == 'ssim':
                   matrix = self.calculate_distance_dataset_ssim(self.dict_datasets[dataset_name], is_rgb=is_rgb, winsize=winsize)
                   self.dict_dataset_description[dataset_name] = self.dict_dataset_description[dataset_name] + f', matrix_nane:{martix_name}'
         if martix_name == 'mse':
                    matrix = self.calculate_distance_dataset_mse(self.dict_datasets[dataset_name], is_rgb)
                    self.dict_dataset_description[dataset_name] = self.dict_dataset_description[dataset_name] + f', matrix_nane:{martix_name}'

         if martix_name == 'wd':
                    matrix = self.calculate_distance_dataset_wd(self.dict_datasets[dataset_name], is_rgb)
                    self.dict_dataset_description[dataset_name] = self.dict_dataset_description[dataset_name] + f', matrix_nane:{martix_name}'

         else: 
                    matrix = self.calculate_distance_dataset_ssim(self.dict_datasets[dataset_name], is_rgb=is_rgb, winsize=winsize)
                    self.dict_dataset_description[dataset_name] = self.dict_dataset_description[dataset_name] + f', matrix_nane:{martix_name}'


         self.dict_matrices[dataset_name] = matrix

    def calculate_distribution_merged_dataset(self, dataset_names, martix_name='ssim', is_rgb=True, channels = 3, winsize=5):
         dataset_name =""
         for name in dataset_names:
               dataset_name+=name
         matrix = []
         if martix_name == 'ssim':
                   matrix = self.calculate_distance_dataset_ssim(self.dict_datasets[dataset_name], is_rgb=is_rgb, winsize=winsize)
                   self.dict_dataset_description[dataset_name] = self.dict_dataset_description[dataset_name] + f', matrix_nane:{martix_name}'
         if martix_name == 'mse':
                    matrix = self.calculate_distance_dataset_mse(self.dict_datasets[dataset_name], is_rgb)
                    self.dict_dataset_description[dataset_name] = self.dict_dataset_description[dataset_name] + f', matrix_nane:{martix_name}'

         if martix_name == 'wd':
                    matrix = self.calculate_distance_dataset_wd(self.dict_datasets[dataset_name], is_rgb)
                    self.dict_dataset_description[dataset_name] = self.dict_dataset_description[dataset_name] + f', matrix_nane:{martix_name}'

         else: 
                    matrix = self.calculate_distance_dataset_ssim(self.dict_datasets[dataset_name], is_rgb=is_rgb, winsize=winsize)
                    self.dict_dataset_description[dataset_name] = self.dict_dataset_description[dataset_name] + f', matrix_nane:{martix_name}'


         self.dict_matrices[dataset_name] = matrix

    def calculate_distribution_datasets(self, dataset_name1 , dataset_name2, martix_name='ssim', is_rgb=True,channels = 3, winsize=5):
         matrix = []
         if martix_name == 'ssim':
                   matrix = self.calculate_distance_datasets_ssim(self.dict_datasets[dataset_name1], self.dict_datasets[dataset_name2], is_rgb=is_rgb, winsize=winsize)
         if martix_name ==  'mse':
                    matrix = self.calculate_distance_datasets_mse(self.dict_datasets[dataset_name1], self.dict_datasets[dataset_name2], is_rgb)

         if martix_name == 'wd':
                    matrix = self.calculate_distance_datasets_wd(self.dict_datasets[dataset_name1], self.dict_datasets[dataset_name2], is_rgb)

         else: 
                matrix = self.calculate_distance_datasets_ssim(self.dict_datasets[dataset_name1], self.dict_datasets[dataset_name2], is_rgb=is_rgb, winsize=winsize)



         self.dict_matrices[dataset_name1+"_cross_"+dataset_name2] = matrix

    def plotCrossDistribution(self, dataset_name1, dataset_name2, title, normalize=True):
         description =""
         y_start=-0.015
        #description = description+'\n'+self.dict_dataset_description[dataset_names[i]]
         x, p = self.plotNormalDistribution(self.dict_matrices[dataset_name1+"_cross_"+dataset_name2], 'k', normalize=normalize)
         plt.plot(x, p, 'k', linewidth=2)
         plt.title(f'Normal Distribution(Between Two Datasets): {title}')
         plt.legend()
         #plt.figtext(1.05, 0.5, description, fontsize=12, va='center', ha='left')
         plt.savefig(f'{title}_{time.time()}_.png', bbox_inches='tight')

         #plt.show()
         plt.clf()


    def plotMergedDistributions(self, dataset_names, title, normalize=True, skip_desc=False):
         dataset_name =""
         for name in dataset_names:
               dataset_name+=name
         description =""
         y_start=-0.015
         

        #description = description+'\n'+self.dict_dataset_description[dataset_names[i]]
         x, p = self.plotNormalDistribution(self.dict_matrices[dataset_name], self.dict_colors[dataset_name], normalize=normalize)
         
         if skip_desc == False:
            plt.plot(x, p, self.dict_colors[dataset_name], linewidth=2, label=dataset_name)
            plt.text(0.8, y_start, self.dict_dataset_description[dataset_name], fontsize=9, color=self.dict_colors[dataset_name], va='center', ha='left', weight='bold')
            y_start-=0.02
         else: 
            plt.plot(x, p, self.dict_colors[dataset_name], linewidth=2)

         plt.title(f'Normal Distribution: {title}')
         plt.legend()
         #plt.figtext(1.05, 0.5, description, fontsize=12, va='center', ha='left')
         plt.savefig(f'{title}_{time.time()}_.png', bbox_inches='tight')

         #plt.show()
         plt.clf()
         
    def plotDistributions(self, dataset_names, title, normalize=True, skip_desc=False):
         description =""
         y_start=-0.015
         for i in range(len(dataset_names)):
              #description = description+'\n'+self.dict_dataset_description[dataset_names[i]]
              x, p = self.plotNormalDistribution(self.dict_matrices[dataset_names[i]], self.dict_colors[dataset_names[i]], normalize=normalize)
              if skip_desc == False:
                plt.plot(x, p, self.dict_colors[dataset_names[i]], linewidth=2, label=dataset_names[i])
                plt.text(0.8, y_start, self.dict_dataset_description[dataset_names[i]], fontsize=9, color=self.dict_colors[dataset_names[i]], va='center', ha='left', weight='bold')
                y_start-=0.02
              else: 
                plt.plot(x, p, self.dict_colors[dataset_names[i]], linewidth=2)
    
         plt.title(f'Normal Distribution: {title}')
         plt.legend()
         #plt.figtext(1.05, 0.5, description, fontsize=12, va='center', ha='left')
         plt.savefig(f'{title}_{time.time()}_.png', bbox_inches='tight')
         #plt.show()
         plt.clf()

    def plotAllDistributions(self, title, normalize=True, skip_desc=False):
         description =""
         y_start=-0.02
         for key in self.dict_datasets.keys():
       
              x, p = self.plotNormalDistribution(self.dict_matrices[key], self.dict_colors[key], normalize=normalize)
              description = description+'\n'+self.dict_dataset_description[key]
              if skip_desc == False:
                plt.plot(x, p, self.dict_colors[key], linewidth=2, label=key)
                plt.text(0.8, y_start, self.dict_dataset_description[key], fontsize=9, color=self.dict_colors[key], va='center', ha='left', weight='bold')
                y_start-=0.02
              else: 
                plt.plot(x, p, self.dict_colors[key], linewidth=2)

    
         plt.title(f'Normal Distribution: {title}')
         plt.legend()
         #plt.figtext(1.05, 0.5, description, fontsize=12, va='center', ha='left')
         plt.savefig(f'{title}_{time.time()}_.png', bbox_inches='tight')
         #plt.show()
         plt.clf()


    #############SSIM###############
    def calculate_distance_dataset_ssim(self, data, is_rgb, winsize):
        len_data = len(data)
        break_point = len_data // 2 if len_data % 2 == 0 else (len_data // 2) + 1
        distance_matrix = []
        
        def calculate_distance(i):
            distance_row = []
            for j in range(len_data):
              try:    
                    euclid_distance=np.nan
                    if i<j:                   
                                ssim_score = 0
                                if is_rgb:
                                    for channel in range(3): 
                                        score, _ = ssim(data[i][:,:,channel], data[j][:,:,channel], full=True, win_size=winsize)
                                        ssim_score += (score/3)
                                
                                else: 
                                    ssim_score, _ = ssim(data[i], data[j], full=True, win_size=winsize)
                                    
                                euclid_distance = 1 - ssim_score
                    
                   
                    distance_row.append(euclid_distance)
              except Exception as e:
                    if is_rgb:
                        try:
                            im1 = rgb2gray(data[i]) if len(data[i].shape) == 3 and (data[i].shape[0] == 3 or data[i].shape[2] == 3) else  rgb2gray( np.repeat(data[i][:, :, np.newaxis], 3, axis=2))
                            im2 = rgb2gray(data[j]) if len(data[j].shape) == 3 and (data[j].shape[0] == 3 or data[j].shape[2] == 3) else  rgb2gray( np.repeat(data[j][:, :, np.newaxis], 3, axis=2))
                            #print(im1.shape, im2.shape)
                            im1 = im1 * 255
                            im2 = im2 * 255
                            
                            #print(im2)
                            #print(data[j])
                            ssim_score, _ = ssim(im1.astype(np.uint8), im2.astype(np.uint8), full=True, win_size=winsize)
                            euclid_distance= 1- ssim_score
                            distance_row.append(euclid_distance)
                        except Exception as e1:
                            print(f'Exception2 with img {i}, {j}, is_rgb: {is_rgb}, type_i: {type(data[i])}, dim_i: {len(data[i].shape)}, shape_i:{data[i].shape}, type_j:{type(data[j])}, dim_j:{len(data[j].shape)}, shape_j:{data[j].shape}', e)
                            distance_row.append(np.nan)

                    else: 
                        print(f'Exception1 with img {i}, {j}, is_rgb: {is_rgb}, type_i: {type(data[i])}, dim_i: {len(data[i].shape)}, shape_i:{data[i].shape}, type_j:{type(data[j])}, dim_j:{len(data[j].shape)}, shape_j:{data[j].shape}', e)
                        distance_row.append(np.nan)
                    continue
            return distance_row
            
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_results = [executor.submit(calculate_distance, i) for i in range(break_point)]
            for future in concurrent.futures.as_completed(future_results):
                distance_matrix.append(future.result())
        return distance_matrix

    def calculate_distance_datasets_ssim(self, data1, data2, is_rgb, winsize):
        len_data1 = len(data1)
        len_data2 = len(data2)
        distance_matrix = []
        def calculate_distance(i):
            distance_row = []
            for j in range(len_data2):
                try:    
                    ssim_score = 0 
                    if is_rgb:
                        for channel in range(3): 
                            score, _ = ssim(data1[i][:,:,channel], data2[j][:,:,channel], full=True, win_size=winsize)
                            ssim_score += (score/3)
                    else:    
                        ssim_score, _ = ssim(data1[i], data2[j], full=True, win_size=winsize)
                            
                    euclid_distance = 1 - ssim_score
                    distance_row.append(euclid_distance) 
                except Exception as e:
                      if is_rgb:
                        try:
                            im1 = rgb2gray(data1[i]) if len(data1[i].shape) == 3 and (data1[i].shape[0] == 3 or data1[i].shape[2] == 3) else  rgb2gray( np.repeat(data1[i][:, :, np.newaxis], 3, axis=2))
                            im2 = rgb2gray(data2[j]) if len(data2[j].shape) == 3 and (data2[j].shape[0] == 3 or data2[j].shape[2] == 3) else  rgb2gray( np.repeat(data2[j][:, :, np.newaxis], 3, axis=2))
                            im1 = im1 * 255
                            im2 = im2 * 255
                            ssim_score, _ = ssim(im1.astype(np.uint8), im2.astype(np.uint8), full=True, win_size=winsize)
                            euclid_distance= 1- ssim_score
                            distance_row.append(euclid_distance)  
                        except:
                                print(f'Exception2 with img {i}, {j}, is_rgb: {is_rgb}, type_i: {type(data1[i])}, dim_i: {len(data1[i].shape)}, shape_i:{data1[i].shape}, type_j:{type(data2[j])}, dim_j:{len(data2[j].shape)}, shape_j:{data2[j].shape}', e)
                                distance_row.append(np.nan)

                      else:
                        distance_row.append(np.nan)
                        print(f'Exception1 with img {i}, {j}, is_rgb: {is_rgb}, type_i: {type(data1[i])}, dim_i: {len(data1[i].shape)}, shape_i:{data1[i].shape}, type_j:{type(data2[j])}, dim_j:{len(data2[j].shape)}, shape_j:{data2[j].shape}', e)
                      continue
                
            return distance_row



        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_results = [executor.submit(calculate_distance, i) for i in range(len_data1)]
            for future in concurrent.futures.as_completed(future_results):
                distance_matrix.append(future.result())
        return distance_matrix
    


    #############MSE###############
    def calculate_distance_dataset_mse(self, data, is_rgb):
        len_data = len(data)
        break_point = len_data // 2 if len_data % 2 == 0 else (len_data // 2) + 1
        distance_matrix = []
        def calculate_distance(i):
            distance_row = []
            for j in range(len_data):
              try:    
                    mse = np.nan
                    if i<j:                   
                                mse = 0
                                if is_rgb:
                                    for channel in range(3): 
                                        mse = mean_squared_error(data[i][:, :, channel], data[j][:, :, channel])
                                        mse += (mse/3)
                                
                                else: 
                                    mse = mean_squared_error(data[i], data[j])
                                    
                            
                    
                   
                    distance_row.append(mse)
              except Exception as e:
                    if is_rgb:
                        im1 = rgb2gray(data[i]) if len(data[i].shape) == 3 and (data[i].shape[0] == 3 or data[i].shape[2] == 3) else  rgb2gray( np.repeat(data[i][:, :, np.newaxis], 3, axis=2))
                        im2 = rgb2gray(data[j]) if len(data[j].shape) == 3 and (data[j].shape[0] == 3 or data[j].shape[2] == 3) else  rgb2gray( np.repeat(data[j][:, :, np.newaxis], 3, axis=2))
                        im1 = (im1 * 255).astype(np.uint8)
                        im2 = (im2 * 255).astype(np.uint8)
                        mse = mean_squared_error(im1, im2)
                        distance_row.append(mse)
                    else: 
                        print(f'Exception with img {i}, {j}, is_rgb{is_rgb}', e)
                        distance_row.append(np.nan)
                    
                    continue
            return distance_row
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_results = [executor.submit(calculate_distance, i) for i in range(break_point)]
            for future in concurrent.futures.as_completed(future_results):
                distance_matrix.append(future.result())

        return distance_matrix

    def calculate_distance_datasets_mse(self, data1, data2, is_rgb):
        len_data1 = len(data1)
        len_data2 = len(data2)
        distance_matrix = []
        def calculate_distance(i):
            distance_row = []
            for j in range(len_data2):
                try:    
                    mse = 0 
                    if is_rgb:
                        for channel in range(3): 
                            mse = mean_squared_error(data1[i][:, :, channel], data2[j][:, :, channel])
                            mse += (mse/3)
                    else:    
                            mse = mean_squared_error(data1[i], data2[j])

                            
                    
                    distance_row.append(mse) 
                except Exception as e:
                      if is_rgb:
                        im1 = rgb2gray(data1[i]) if len(data1[i].shape) == 3 and (data1[i].shape[0] == 3 or data1[i].shape[2] == 3) else  rgb2gray( np.repeat(data1[i][:, :, np.newaxis], 3, axis=2))
                        im2 = rgb2gray(data2[j]) if len(data2[j].shape) == 3 and (data2[j].shape[0] == 3 or data2[j].shape[2] == 3) else  rgb2gray( np.repeat(data2[j][:, :, np.newaxis], 3, axis=2))
                        im1 = (im1 * 255).astype(np.uint8)
                        im2 = (im2 * 255).astype(np.uint8)
                        mse = mean_squared_error(im1, im2)
                        distance_row.append(mse)  
                      else:
                        print(f'Exception with img {i}, {j}, is_rgb{is_rgb}', e)
                        distance_row.append(np.nan)
                      continue
                
            return distance_row
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_results = [executor.submit(calculate_distance, i) for i in range(len_data1)]
            for future in concurrent.futures.as_completed(future_results):
                distance_matrix.append(future.result())
        return distance_matrix
    
    #############wasserstein_distance###############
    def image_to_histogram(self, image):
        hist_r, _ = np.histogram(image[:,:,0].flatten(), bins=256, range=(0,256))
        hist_g, _ = np.histogram(image[:,:,1].flatten(), bins=256, range=(0,256))
        hist_b, _ = np.histogram(image[:,:,2].flatten(), bins=256, range=(0,256))
        
        hist = np.concatenate((hist_r, hist_g, hist_b))
        return hist.astype(float)

    def calculate_distance_dataset_wd(self, data, is_rgb):
        len_data = len(data)
        break_point = len_data // 2 if len_data % 2 == 0 else (len_data // 2) + 1
        distance_matrix = []
        def calculate_distance(i):
            distance_row = []
            for j in range(len_data):
              try:    
                    w_distance =np.nan
                    if i<j:                   
                            
                                if is_rgb:
                                    hist1 = self.image_to_histogram(data[i])
                                    hist2 = self.image_to_histogram(data[j])
                                
                                else: 
                                    hist1, _ = np.histogram(data[i].flatten(), bins=256, range=(0,256))
                                    hist2, _ = np.histogram(data[j].flatten(), bins=256, range=(0,256))
                                w_distance =  wasserstein_distance(hist1, hist2)
                    
                   
                    distance_row.append(w_distance)
              except Exception as e:
                    if is_rgb:
                        hist1, _ = np.histogram(data[i].flatten(), bins=256, range=(0,256))
                        hist2, _ = np.histogram(data[j].flatten(), bins=256, range=(0,256))
                        w_distance =  wasserstein_distance(hist1, hist2)
                        distance_row.append(w_distance)
                    else: 
                        distance_row.append(np.nan)
                        print(f'Exception with img {i}, {j}, is_rgb{is_rgb}', e)
                    continue
            return distance_row
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_results = [executor.submit(calculate_distance, i) for i in range(break_point)]
            for future in concurrent.futures.as_completed(future_results):
                distance_matrix.append(future.result())
        return distance_matrix

    def calculate_distance_datasets_wd(self, data1, data2, is_rgb):
        len_data1 = len(data1)
        len_data2 = len(data2)
        distance_matrix = []
        def calculate_distance(i):
            distance_row = []
            for j in range(len_data2):
                try:    
                    if is_rgb:
                        hist1 = self.image_to_histogram(data1[i])
                        hist2 = self.image_to_histogram(data2[j])
                    else:    
                        hist1, _ = np.histogram(data1[i].flatten(), bins=256, range=(0,256))
                        hist2, _ = np.histogram(data2[j].flatten(), bins=256, range=(0,256))
                    w_distance =  wasserstein_distance(hist1, hist2)
                    distance_row.append(w_distance)
                except Exception as e:
                      if is_rgb:
                        hist1, _ = np.histogram(data1[i].flatten(), bins=256, range=(0,256))
                        hist2, _ = np.histogram(data2[j].flatten(), bins=256, range=(0,256))
                        w_distance =  wasserstein_distance(hist1, hist2)
                        distance_row.append(w_distance)
                      else:
                        distance_row.append(np.nan)
                        print(f'Exception with img {i}, {j}, is_rgb{is_rgb}', e)
                      continue
                
            return distance_row

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_results = [executor.submit(calculate_distance, i) for i in range(len_data1)]
            for future in concurrent.futures.as_completed(future_results):
                distance_matrix.append(future.result())
        return distance_matrix
    def display_intersectionmatrix(self, intersection_matrix, xlabel, ylabel):
        intersection_matrix = np.nan_to_num(intersection_matrix, nan=0)
        fig, ax = plt.subplots()
        ax.matshow(intersection_matrix, cmap=plt.cm.Blues)
        x_len = len(intersection_matrix)
        y_len = len(intersection_matrix[0])
        
        for i in range(y_len):
            for j in range(x_len):
                c = intersection_matrix[j][i]
                c = int((c - int(c))*10)
                ax.text(i, j, str(c), va='center', ha='center')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.show()
        plt.clf()

    def plotNormalDistribution(self, intersection_matrix, color='k', normalize=False):
        interesection_values = np.ravel(intersection_matrix)
        nan_mask = np.isnan(interesection_values)
        interesection_values = interesection_values[~nan_mask]


        if normalize:
            min_value = np.min(interesection_values)
            max_value = np.max(interesection_values)
            normalized_data = (interesection_values - min_value) / (max_value - min_value)
            mean, std_dev = norm.fit(normalized_data)
            plt.hist(normalized_data, weights=np.ones(len(normalized_data)) / len(normalized_data), bins=20, alpha=0.6, color=color, edgecolor='black', histtype='stepfilled')
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mean, std_dev)/100
        
        else:
            mean, std_dev = norm.fit(interesection_values)
            plt.hist(interesection_values,  bins=20, alpha=0.6, color=color, edgecolor='black', histtype='stepfilled')
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mean, std_dev)
        
        return x, p
    
    
    


def calculate_distribution(title, dist_class, dist_class_path, include_grey=False, is_rgb=True, m_name='ssim'):
    anomaly_class_names = os.listdir(dist_class_path+'/test')
    anomaly_class_names.remove('good')
    anomaly_class_merged = ''
    for i in range(0, len(anomaly_class_names)):
        anomaly_class_merged += f'{dist_class}_test_{anomaly_class_names[i]}'
        anomaly_class_names[i] = f'{dist_class}_test_{anomaly_class_names[i]}'

    #print(anomaly_class_names, anomaly_class_merged)
    resizeH = 256
    resizeW=256
    obj =  CalculateImageDistribution()
    print('################', obj.mvtech_train_len)
    obj.set_mvtech_dataset(dist_class_path, dataset_name=dist_class, load_all=True, datasetSize=10, divisions=4, resizeH=resizeH, resizeW=resizeW, include_grey=include_grey)
    print('################', obj.mvtech_train_len)
    if obj.mvtech_train_len == 0: 
            obj.merge_datasets(anomaly_class_names)
            obj.calculate_distribution_merged_dataset(anomaly_class_names, martix_name = m_name, is_rgb=is_rgb)
            obj.plotMergedDistributions(anomaly_class_names, f'{dist_class} Anomoly Merged')
            obj.save_dicts(f'{dist_class}_{resizeH}_x_{resizeH}')
            return 

    obj.calculate_all_distributions(matrix_name = m_name, is_rgb=is_rgb)
    #obj.plotAllDistributions(f'{title}')


    obj.merge_datasets(anomaly_class_names)
    obj.calculate_distribution_merged_dataset(anomaly_class_names, martix_name = m_name, is_rgb=is_rgb)


    obj.plotDistributions([f'{dist_class}_train'], f'{title} train', skip_desc=True)
    obj.plotDistributions([f'{dist_class}_test_good'], f'{title} test good', skip_desc=True)
    for i in range(0, len(anomaly_class_names)):
        obj.plotDistributions([f'{anomaly_class_names[i]}'], f'{anomaly_class_names[i]}', skip_desc=True)


    obj.plotMergedDistributions(anomaly_class_names, f'{dist_class} Anomoly Merged', skip_desc=True)


    obj.calculate_distribution_datasets(f'{dist_class}_train', anomaly_class_merged, martix_name = m_name, is_rgb=is_rgb)
    obj.plotCrossDistribution(f'{dist_class}_train', anomaly_class_merged, f'{dist_class} Train vs Anomaly Merged Cross Distribution')
    obj.save_dicts(f'{dist_class}_{resizeH}_x_{resizeH}')



def list_folders(path):
    folders = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    return folders

mvtec_path = '/scratch/cluster/data/mvtec'
list_mvtec = list_folders(mvtec_path)
grey_class = ['grid','screw', 'zipper']
for target_class in list_mvtec:
     title = target_class[:1].upper() + target_class[1:]
     dist_class = target_class
     dist_class_path = os.path.join(mvtec_path, dist_class)
     if dist_class in grey_class:
          calculate_distribution(title, dist_class, dist_class_path, include_grey=True, is_rgb=False, m_name='ssim')
     else: 
          calculate_distribution(title, dist_class, dist_class_path, include_grey=True, is_rgb=True, m_name='ssim')  


