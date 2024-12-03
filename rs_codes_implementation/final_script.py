import glob
import CZDS_utils
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import os 

from sklearn.linear_model import LinearRegression

#

# Definir a pasta que contém os arquivos .xdi
#pasta = "teste_xdi" #(testes)
pasta = "XDIs" #Todos os .xdi da base

# Usar o glob para pegar todos os arquivos .xdi dentro da pasta
arquivos = glob.glob(f"{pasta}/*.xdi")

# Loop para processar cada arquivo na pasta
for xdi in arquivos:
    print(f"Processando o arquivo: {xdi}")

    #criar lista vazia 
    
    #xdi = "XDIs/as2o5_10K_scan1.xdi"

    data = CZDS_utils.read.xdi(xdi)
    #print (data) 

    if data[0] == 'Transmission':
        if CZDS_utils.XASNormalization.xas_type(data[1]) == "EXAFS":
            enorm, mu_norm = CZDS_utils.XASNormalization.EXAFS_normalization(data[1], data[4], debug=False)
            plt.scatter(enorm, mu_norm)
            plt.show()
        else:
            enorm, mu_norm = CZDS_utils.XASNormalization.XANES_normalization(data[1], data[4], debug=False)
            plt.scatter(enorm, mu_norm, label='Normalized')
            plt.show()
    elif data[0] == 'Transmission Raw':
        if CZDS_utils.XASNormalization.xas_type(data[1]) == "EXAFS":
            #tranmission_raw_exafs_counter += 1
            enorm, mu_norm = CZDS_utils.XASNormalization.EXAFS_normalization(data[1], data[2], debug=False)
            plt.scatter(enorm, mu_norm)
            plt.show()
        else:
            enorm, mu_norm = CZDS_utils.XASNormalization.XANES_normalization(data[1], data[2], debug=False)
            plt.scatter(enorm, mu_norm, label='Normalized')
            plt.show()
    elif data[0] == 'Normalized Transmission':
        if CZDS_utils.XASNormalization.xas_type(data[1]) == "EXAFS":
            enorm, mu_norm = CZDS_utils.XASNormalization.EXAFS_normalization(data[1], data[2], debug=False)
            plt.scatter(enorm, mu_norm)
            plt.show()
        else:
            enorm, mu_norm = CZDS_utils.XASNormalization.XANES_normalization(data[1], data[2], debug=False)
            plt.scatter(enorm, mu_norm)
            plt.show()
    elif data[0] == 'Fluorescence':
        #fluorescence_counter += 1
        if CZDS_utils.XASNormalization.xas_type(data[1]) == "EXAFS":
            enorm, mu_norm = CZDS_utils.XASNormalization.EXAFS_normalization(data[1], data[2], debug=False)
            plt.scatter(enorm, mu_norm)
            plt.show()
        else:
            enorm, mu_norm = CZDS_utils.XASNormalization.XANES_normalization(data[1], data[2], debug=False)
            plt.scatter(enorm, mu_norm, label='Normalized')
            plt.show()

    #ORIGINAL:

    #print(xdi)
    filename = xdi.replace('.xdi', '')
    
    #filename = filename.replace('XDIs/', '') # para quando estiver usando todos os arquivos xdi
    filename = filename.replace('teste_xdi/', '') # para quando estiver usando o teste.xdi
    
    print(f'O filename:{filename}')

    #

    #print(enorm)
    #print(mu_norm)

    # 

    #Creates a dataframe where the keys are the names of columns and the values, the lines.
    df = pd.DataFrame({'energy': enorm, filename : mu_norm})
    #print(df) 

    #Acess what fold are the .csv archives
    way = 'XANES-csv/teste_xdi.csv'
    print(f'O csv foi salvo no arquivo: {way}')
    
    #Transform the pandas dataframe into .csv
    df.to_csv(way, index = False)


# 

    def calculate_derivatives(csv_file_path, column_x, column_y, output_1st_deriv, output_2nd_deriv):
        """
        Function to calculate the first and second derivatives of one column with respect to another and save the results in CSV files.

        Parameters:
        - csv_file_path: path to the input CSV file.
        - column_x: name of the column representing the x-axis.
        - column_y: name of the column representing the y-axis (for derivative calculation).
        - output_1st_deriv: path to save the CSV file with the first derivative.
        - output_2nd_deriv: path to save the CSV file with the second derivative.
        """
        # Reading the CSV file
        df = pd.read_csv(csv_file_path)

        # Calculating the derivative of y with respect to x
        df['dy_dx'] = np.gradient(df[column_y], df[column_x]) 

        # Calculating the second derivative
        df['dy2_dx2'] = np.gradient(df['dy_dx'], df[column_x])

        # Creating new DataFrames for the first and second derivatives
        df_1st_deriv = df[[column_x, 'dy_dx']]
        df_2nd_deriv = df[[column_x, 'dy2_dx2']]

        # Exporting to CSV
        df_1st_deriv.to_csv(output_1st_deriv, index=False)
        df_2nd_deriv.to_csv(output_2nd_deriv, index=False)

        return df

    # 

    #calculate_derivatives(way, 'energy', filename, 'XANES-derivatives/1st_deriv.csv', 'XANES-derivatives/2nd_deriv.csv')
    calculate_derivatives(way, 'energy', filename, f'XANES-derivatives/{filename}_1st_deriv.csv', f'XANES-derivatives/{filename}_2nd_deriv.csv') # salvando derivadas em locais arquivos diferentes! Deu certo!! 
    
    #

    ## Packages
    import csv
    import numpy as np
    #import seaborn
    #%matplotlib inline
    import matplotlib.pyplot as plt
    ### To do normalization
    from sklearn.preprocessing import normalize
    import pandas as pd
    import multiprocessing 
    import time

    ### To do normalization
    from sklearn.preprocessing import normalize

    # To prevent weird, long warning
    import warnings
    warnings.filterwarnings(action="ignore",message="internal issue")

    # where we are saving stuff
    import os
    PROJECT_DIR =  os.getcwd()#os.path.dirname(os.path.realpath(__file__))
    NOTEBOOK = "XANES_figures"
    IMAGES = os.path.join(PROJECT_DIR,"figures",NOTEBOOK)

    ## A save figures function
    def figure_save(figure_name, tight_layout=True, figure_extension="eps",resolution=300):
        path = os.path.join(IMAGES,figure_name.replace(" ", "_")+"."+figure_extension)
        print("Saving figure as ",figure_name.replace(" ", "_"))
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path,format=figure_extension,dpi=resolution)

    # 

    ## For Fe data --- List all the csv files in the directory
    import glob 
    what_files = glob.glob('XANES-csv/*.csv')

    data=pd.DataFrame([])

    for name_of_file in what_files:
        # read  first row with name of components
        aux = pd.read_csv(name_of_file,delimiter=',') 

        ## Concatenate them all
        data = pd.concat([data,aux],axis=1)

    # 

    names = list(data.columns)[1:]
    ### These are the dictionaries where we will save numpy arrays
    raw_materials, E, A={}, {}, {}

    data_val = data.values
    numb_files = len(what_files)

    for i in range(1,data.shape[1]):
        name = names[i-1]
        E[name] = np.reshape(data_val[:,0],(1,-1))
        raw_materials[name] = np.reshape(data_val[:,i],(1,-1))  

    # 

    #print(name)
    #names

    #

    L = len(names)
    print (f' Esse é o L: {L}')
    print (f' Esse é o names: {names} e esse é o filename:{filename}')
    print (f' Esse é o names[0]: {names[0]}')
    
    plt.figure(figsize=(15,8))
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    for i in range(0,L):
        plt.plot(E[names[i]].T,raw_materials[names[i]].T,color='C'+str(i%10), label = str(names[i]),lw=4,linestyle='-')

    plt.title('Xanes spectrum',size=28)
    plt.ylabel('$\mu(E)$',size=28)
    plt.xlabel("Energy (eV)", size=28)
    plt.legend(loc=4,prop={'size':22})
    plt.grid(True)
    #figure_save("XANES_some_examples"+names[0])
    plt.show()     

    # 

    #what_files_derivatives = glob.glob(PROJECT_DIR+"/xafsderivative/Fe_1st_derivative.csv")
    what_files_derivatives = glob.glob(PROJECT_DIR+"/XANES-derivatives/1st_deriv.csv")#

    data_derivatives=pd.DataFrame([])

    for name_of_file in what_files_derivatives:
        # read  first row with name of components
        aux = pd.read_csv(name_of_file,delimiter=',') 

        ## Concatenate them all
        data_derivatives = pd.concat([data_derivatives,aux],axis=1)


    names_1st_derivatives = list(data_derivatives.columns)[1:]
    ### These are the dictionaries where we will save numpy arrays
    raw_materials_1st_derivatives, E_derivatives, A_derivatives={}, {}, {}
    data_val_derivatives = data_derivatives.values 

    for i in range(1,data_derivatives.shape[1]):
        name = names_1st_derivatives[i-1]
        E_derivatives[name] = np.reshape(data_val_derivatives[:,0],(1,-1))
        raw_materials_1st_derivatives[name] = np.reshape(data_val_derivatives[:,i],(1,-1))    

    # 

    #print(data_derivatives.columns)

    # 

    what_files_derivatives = glob.glob(PROJECT_DIR+"/derivatives/2nd_deriv.csv")

    data_derivatives=pd.DataFrame([])

    for name_of_file in what_files_derivatives:
        # read  first row with name of components
        aux = pd.read_csv(name_of_file,delimiter=',') 

        ## Concatenate them all
        data_derivatives = pd.concat([data_derivatives,aux],axis=1)

    ### These are the dictionaries where we will save numpy arrays
    raw_materials_2nd_derivatives, E_derivatives, A_derivatives={}, {}, {}
    data_val_derivatives = data_derivatives.values

    for i in range(1,data_derivatives.shape[1]):
        name = names_1st_derivatives[i-1]
        E_derivatives[name] = np.reshape(data_val_derivatives[:,0],(1,-1))
        raw_materials_2nd_derivatives[name] = np.reshape(data_val_derivatives[:,i],(1,-1))    

    # 

    #print(raw_materials_2nd_derivatives.keys())

    # 
    def normalization(materials,names,normalizeheight=False):
        '''
        This function returns  a vertically translated, and possibly stretched vertically, copy of
        the a spectrum to  a vector X, in such a way that 

        (i) X[0,0] = 0
        holds     and

        (ii) max(X) - min(X)=1  holds when normalizeheight = True
        ----------------
        Input:

        -names: vector of strings
        -materials: dictionary with spectra, indexed by elements in vector names
        -normalizeheight: a boolean entry, that defines whether the normalization (**) happens
        ----------------
        Output is/are:
        - X: the renormalized vector
        - shift_height: the amout we had to shift the vectoverticaly in order to have X[0.0]=0 
        ----------------
        '''
        shift_height, normalization_heights, X={}, {}, {}
        for name in names:
            shift_height[name] = np.copy(materials[name][0,0])
            X[name]  = materials[name]- shift_height[name]

        if normalizeheight:
            ### Now we use the function normalize, from sklearn
            for name in names:
                X[name],normalization_heights[name] = normalize(X[name], norm='max', axis=1,return_norm=True)

        return X, shift_height, normalization_heights

    #########################################################################################################
    def oscillation_function(interval,f,full_computation=True):
        ''' set an oscillation function f on an interval interval.

        ----------------
        Input:

        - interval of computation: THIS IS NOT NECESSARY, ACTUALLY!
        - f: the function whose oscillation will be computed
        - full_computation, a flag, where user define whether to compute fulll oscillation function, 
          or just the threshold

        ----------------
        Output:

        - x: a vector labeled as 1...N
        - oscillation: oscillation of the function f
        - estimated oscillation: the threshold oscillation, given as oscillation[0,1]
        ----------------
        '''
        #plot the oscillation of the function
        L = interval.shape[1]
        x = np.reshape(np.arange(L),(1,-1))
        oscillation = np.zeros([1,L]).reshape(1,-1)

        if full_computation:
            for l in range(1,L):
                oscillation[0,l] = np.max([np.max(f[0,k:k+l+1]) - np.min(f[0,k:k+l+1]) for k in range(L-l)])

            return (x, oscillation, oscillation[0,1])
        else:
            oscil_number = np.max([np.max(f[0,k:k+1+1]) - np.min(f[0,k:k+1+1]) for k in range(L-1)])

            return (x, None, oscil_number)
    #########################################################################################################    
    def plateau_detection (v,L_threshold):
        '''
        This function detects plateaus of length > L_threshold in a vector v. 
        ----------------
        Input:
        - v: a matrix with shape 1 X m
        - L_threshold: plateaus with length L < L_threshold will not be considered]. L_threshold >=2
        ----------------
        Output:

        It returns their 
        - location: the entry of the peak in the vector
        - the value that the vector v take at that point.

        If no peak is found then it returns (None, None)
        ----------------
        '''
        v_temp = np.copy(v).reshape(1,-1) # this is a row vector, not a rank one vector

        # In order to detect the plateaus, we make a copy of the vector and compare it with a shifted version of itself
        v_aux =np.copy(v_temp[0,0:-1]) ## a smaller version of v
        v_shifted = np.copy(v_temp[0,1:])
        ''' They are going to be use in the following fashion:
            anytime we have a number 1 means that the next element is the same to the next one.
            The length of a sequence of 1s defines the length of that sequence'''

        # Now we compare v_aux and v_shifted, not forgetting to add 0's in the beggining and in the end;
        # We do that because we also want to find sequences in the extremes of the vector
        where_plts_start_and_length = np.array(1*(v_aux==v_shifted),ndmin=2)
        where_plts_start_and_length = np.concatenate([[0],where_plts_start_and_length[0],[0]]).reshape(1,-1)

        '''
        NOTE #1: where_plts_start_and_length is avector with schape 1X m+1
        NOTE #2: sequences of 1`s with length a in where_plts... indicate sequnces in v with length a+1

         At this point we define sequences of 010, 0110 etc, and look for them in the vector.
         If the number of 1's is strictly less than L_threshold -1 then we ignore the plateau.
         '''
        for i in range(1,L_threshold-1):  
            # we shall ignore strings that are smaller than L_threshold
            # now we define a list of 1s, with legth i
            sequence = np.ones([1,i],dtype=np.int32)
            # 1- AUGMENTING THE VECTOR WITH ZEROS
            sequence = np.concatenate([[0],sequence[0],[0]]).reshape(1,-1)

            # Now we search for it in the vector where_plts_start_and_length
            range_search = where_plts_start_and_length.shape[1]-sequence.shape[1]+1

            is_there = [j for j in range(range_search) if str(where_plts_start_and_length[0,j:j+sequence.shape[1]]) ==str(sequence.squeeze()) ]      
            if is_there != []:
                for k in is_there :
                    where_plts_start_and_length[0,k:k+sequence.shape[1]] = np.zeros([1,sequence.shape[1]])## erase that small string

        '''
        2- at this point the vector where_plts_start_and_length has no sequence of 1's with length < L_threshold -1
        the first maximum will be at 
        '''

        if (np.sum(where_plts_start_and_length) ==0):
            #print("No peaks!")
            return (None,None)

        Len_where = where_plts_start_and_length.shape[1]-1
        locations = np.array([],np.int16,ndmin=2)

        for i in range(Len_where):
            # Whenever we have a seq of 1s, we know for sure that it has length bigger or equal to L_threshold,
            # so we can return the value of it`s location

            if (where_plts_start_and_length[0,i+1]- where_plts_start_and_length[0,i] ==1):
                locations = np.concatenate((locations,np.array([i],np.int16, ndmin=2)),axis=1) # subtract 1, because the vector was augmented by 0 in the extremes

        locations = np.reshape(locations[0], (1,-1))
        return (locations[0],v[0,locations[0]])


    #########################################################################################################
    def start_from_peak(Maj,L_threshold):
        '''
        Given a majorating function that starts with a plateau, find its second plateau, as long as the 
        latter has length bigger than L_threshold.
        ----------------
        Input:
        -Maj: a rising sun function
        -L_threshold: the length of the plateaus that should be taken into account 
        ----------------
        Output:
        -location: index of the next plateau
        - value: value of the function at the next plateau
        ----------------
        '''

        L = Maj.shape[1];
        m = Maj[0,0];
        i=1;
        while(i<L-2 and m == Maj[0,i]):
                i= i+1;

        locations,value = plateau_detection(np.array(Maj[0,i:],ndmin=2),L_threshold);
        return (locations+i, value)


    #########################################################################################################
    def padded_spectra(E, materials, names):
        '''
        Embeds Energy, XANES measurement spectrum into larger space, padding the Xanes measurement by a constant.
        Returns  emebedded Energy, embedded Xanes measurement, and index
        ----------------
        Input:
        - E: dictionary indexed by names, with energy interval
        -mu: dictionary indexed by names, with XANES measurements
        -names: keys of the previous dictionaries with names of materials

        ----------------
        Output:

        - padded_energy: dictionary indexed by names containing an extended energy vector to the right
        - padded_material: dictionary indexed by names containing a padded (o the right, by constant) xanes measurement
        -stopping: index of the last entry before the padding starts
        ----------------
        '''

        padded_material,  padded_energy, stopping ={}, {}, {}

        for name in names:
            stopping[name]= materials[name].shape[1]
            # the material vector needs to be padded....
            material_aux = np.squeeze(materials[name])
            #... while the energy vector needs to be extended 
            energy_aux = E[name] +E[name][0,-1]+E[name][0,1]-2*E[name][0,0]

            padded_material[name] = np.reshape(np.pad(material_aux,(0,stopping[name]),'edge'),(1,-1))
            energy_aux = np.concatenate((E[name],energy_aux),axis=1)
            padded_energy[name] = np.reshape(energy_aux,(1,-1))

        return padded_energy,padded_material,stopping

    #########################################################################################################

    def small_padded_spectra(E, F):
        '''
        Embeds two vectors E and F into larger space, extending E in one length to the right and
        padding the F measurement by a constant.
        Returns  emebedded Energy, embedded Xanes measurement.

        Unlike the function padded_spectra(E, materials, names), it does not need names
        ----------------
        Input:
        - E: numpy array denoting  energy interval
        - F: numpy array  denoting  XANES measurements
        ----------------
        Output:

        - padded E: numpy array  containing an extended energy vector to the right
        - padded_ F: numpy array  containing a padded (o the right, by constant) xanes measurement
        -stopping: index of the last entry before the padding starts
        ----------------
        '''
        stop= E.shape[1]
        # the material vector needs to be padded....
        F_aux = np.squeeze(F)
            #... while the energy vector needs to be extended 
        energy_aux = E +E[0,-1]+E[0,1]-2*E[0,0]

        padded_F = np.reshape(np.pad(F_aux,(0,stop),'edge'),(1,-1))
        energy_aux = np.concatenate((E,energy_aux),axis=1)
        padded_energy = np.reshape(energy_aux,(1,-1))

        return padded_energy,padded_F,stop

    # 

    def rising_sun(mu):
        '''
        Given L = length of mu, return a vector maxim_mu of length L,for which, at any index 
         0 <=k <L we have  maxim_mu[k] >= maxim_mu[j],for all 0 <=j <k  

        ----------------
        Input:
        -mu: the XANES measurement

        ----------------
        Output:
        - maxim_mu: therising_sun function  associated to mu
        ----------------
         '''
        L = mu.shape[1]
        maxim_mu = np.zeros([1,L])
        maxim_mu[0,0]=mu[0,0]

        for i in range(1,L):
            if mu[0,i]>maxim_mu[0,(i-1)]:
                maxim_mu[0,i] = mu[0,i]
            else:
                maxim_mu[0,i] = maxim_mu[0,i-1]

        return maxim_mu

    # 
    print(names)

    materials, shift_height, normalizer_shift =normalization(raw_materials,names,normalizeheight=True)

    E_padded,materials, stopping = padded_spectra(E,materials,names)

    # 

    #print(names)
    #print(np.shape(E[names[0]]), np.shape(E_padded[names[0]]))

    # 

    parameters ={}

    for name in names:
        parameters[name]={
            # The non-normalized xanes
            'raw_xanes': raw_materials[name],
            'raw_energy': E[name],

            # The normalized xanes and embedded xanes
            'xanes': materials[name],
            'energy': E_padded[name],

            # The normalization and emebdding info
            'stop': stopping[name],
            'shift_height': shift_height[name],
            'normalizer_shift': normalizer_shift[name]
        }

    # 

    hyperparameters={
        ## These hyperparameters will be used in 
        'lambda_h':1,
        'lambda_d':1/2,

        ## These hyperparameters will be used in find_first_peak
        'lambda_h_find_1st':4,
        'lambda_d_find_1st':1/4,

        ## These hyperparameters will be used in 
        'lambda_d_shrink_1st':1/2, 
        'initial_oscillation_guess_parameter':10,

        ## These hyperparameters will be used in 
        'stretching_factor':3,
        'iteration_decay':.9, 

        ## Type of decay_rate
        'decay_rate_type': 'min_max'
    }

    # 

    ### Let's replot, normalizing first
    plt.figure(figsize=(15,7))
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    i=0

    for name in {names[0]}:
        # Unpacking ....
        # ...parameters
        parameters_now= parameters[name]
        xanes_now = parameters_now['xanes']
        E_now =parameters_now['energy']
        xanes_now_unpad = parameters_now['raw_xanes']
        E_now_unpad =parameters_now['raw_energy']
        plt.plot(E_now_unpad.T,xanes_now_unpad.T,color='C0', label = str(names[i])+'non-normalized',lw=4,linestyle='-')
        plt.plot(E_now.T,xanes_now.T,color='C1', label = str(names[i])+'normalized and padded',lw=4,linestyle='-')

        i+=1

    plt.title('Embedded spectrum using padding', size=28)
    plt.ylabel('$\mu(E)$',size=28)
    plt.xlabel("Energy (eV)", size=28)
    plt.legend(loc=4,prop={'size':22})
    plt.grid(True)
    #figure_save("XANES_some_examples_normalized_and_embedded"+names[0])
    plt.show()    

    # 

    L = len(names)

    rising_sun_f ={}
    plt.figure(figsize=(15,10))
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    i=0  # a flag, used to vary the plot colors

    for name in names:
        # Unpacking ....
        # ...parameters
        parameters_now = parameters[name]
        xanes_now = parameters_now['xanes']
        E_now =parameters_now['energy']
        xanes_now_unpad = parameters_now['raw_xanes']
        E_now_unpad =parameters_now['raw_energy']

        rising_sun_f[name] = rising_sun(xanes_now)
        plt.plot(
            E_now.T,rising_sun_f[name].T,color='C'+str(i%10),\
            label = '$\mathscr{R}_{\mu}(\cdot)$ of '+str(name),marker='o',markersize=4,lw=3,linestyle=':')
        plt.plot(E_now.T,xanes_now.T,color='C'+str(i%10), label ='$\mu(\cdot)$ of '+ str(name),lw=3,linestyle='--')
        i+=1

    plt.title('Rising sun function',size=28)
    plt.ylabel('$\mathscr{R}_{\mu}(E)$',size=28)
    plt.xlabel("Energy (eV)", size=28)
    plt.legend(loc=4,prop={'size': 22})
    plt.grid(True)
    #figure_save("XANES_and_rising_sun"+names[0])
    plt.show()    

    # 

    valley_of_shadows = {}

    plt.figure(figsize=(15,10))
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    i=0

    for name in names:
        # Unpacking ....
        # ...parameters
        parameters_now = parameters[name]
        xanes_now = parameters_now['xanes']
        E_now =parameters_now['energy']
        xanes_now_unpad = parameters_now['raw_xanes']
        E_now_unpad =parameters_now['raw_energy']

        valley_of_shadows[name] = rising_sun_f[name]- xanes_now #this function will always be nonnegative, thanks to (1)
        plt.plot(E_now.T,valley_of_shadows[name].T,color='C'+str(i%10), label =name,lw=4,linestyle='-')
        i+=1

    plt.title('Valley of Shadows function',size=28)
    plt.legend(loc=4,prop={'size': 22})
    plt.ylabel(r"$\mathcal{V}_S[f]$",size=28)
    plt.xlabel('Energy (eV)',size=28)
    plt.grid(True)
    #figure_save("valley_of_shadows"+names[0])
    plt.show()    

    # 


    plt.figure(figsize=(15,10))
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    for name in {names[0]}:
        # Unpacking ....
        # ...parameters
        parameters_now = parameters[name]
        xanes_now = parameters_now['xanes']
        E_now =parameters_now['energy']
        xanes_now_unpad = parameters_now['raw_xanes']
        E_now_unpad =parameters_now['raw_energy']

        plt.plot(E_now[0,0:stopping[name]].T,xanes_now[0,0:stopping[name]].T,\
                 color='C'+str(i%10),lw=4,linestyle='--', label = r"$\mu(\cdot)$: "+ str(name))
        plt.plot(E_now[0,0:stopping[name]].T,rising_sun_f[name][0,0:stopping[name]].T,\
                 color='red',lw=4,linestyle='-', label = r"$\mathcal{R}_{\mu}(\cdot)$ :"+ str(name),markersize=2)
        plt.plot(E_now[0,0:stopping[name]].T,valley_of_shadows[name][0,0:stopping[name]].T,\
                 color='blue',linestyle='-.',lw=4, label = r"$\mathcal{V}_{\mu}(\cdot)$ : "+ str(name))

    plt.title(
        r"The (non-embedded) functions $\mu(\cdot),  \mathcal{V}_{\mu}(\cdot)$ and $\mathcal{R}_{\mu}(\cdot)$",
        size=28)
    plt.ylabel(name+" and\n auxiliar functions",size=28)
    plt.xlabel("Energy (eV)", size=28)

    plt.legend(loc=4,prop={'size': 22})
    plt.grid(True)
    #figure_save("XANES_and_rising_sun"+names[0])
    plt.show()    

    # 

    oscillation, interval, estimate={}, {}, {}

    plt.figure(figsize=(15,8))
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    for name in names:
        # Unpacking ....
        # ...parameters
        parameters_now = parameters[name]
        xanes_now = parameters_now['xanes']
        E_now =parameters_now['energy']
        xanes_now_unpad = parameters_now['raw_xanes']
        E_now_unpad =parameters_now['raw_energy']

        (interval[name],oscillation[name], estimatE_now) = oscillation_function(E_now,xanes_now)
        new_label = name+", noise:" +str(round(estimatE_now,4))
        plt.plot(interval[name].T,oscillation[name].T,linestyle='-',lw=4, label=new_label)

    plt.title('Oscillation function $\omega$',size=28)
    plt.legend(loc=4,prop={'size': 22})
    plt.ylabel('$\omega$',size=28)
    plt.xlabel('Spreading size',size=28)
    plt.grid(True)
    #figure_save("Oscillation")
    plt.show()

    # 

    plt.figure(figsize=(15,8))
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    for name in {names[0]}:
        # Unpacking ....
        # ...parameters
        parameters_now = parameters[name]
        xanes_now = parameters_now['xanes']
        E_now =parameters_now['energy']
        xanes_now_unpad = parameters_now['raw_xanes']
        E_now_unpad =parameters_now['raw_energy']

        (interval[name],oscillation[name], estimatE_now) = oscillation_function(E_now,xanes_now)
        new_label = name+", noise:" +str(round(estimatE_now,4))
        plt.plot(interval[name].T,oscillation[name].T,linestyle='-',lw=4, label=new_label)

    plt.title('Oscillation function $\omega$',size=28)
    plt.legend(loc=4,prop={'size': 22})
    plt.ylabel('$\omega$',size=28)
    plt.xlabel('Spreading size',size=28)
    plt.grid(True)
    #figure_save("Oscillation"+names[0])
    plt.show()

    #


    plt.figure(figsize=(15,8))
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    for name in {names[0]}:
        # Unpacking ....
        # ...parameters
        parameters_now = parameters[name]
        xanes_now = parameters_now['xanes']
        E_now =parameters_now['energy']
        xanes_now_unpad = parameters_now['raw_xanes']
        E_now_unpad =parameters_now['raw_energy']

        f = plt.figure(figsize=(15,7))

        ax1 = plt.subplot(211)
        new_label = name+", noise:" +str(round(estimatE_now,4))
        ax1.plot(interval[name].T,oscillation[name].T,linestyle='-',lw=2, label="$\omega^{\mu}$: "+name,color='r')

        (interval_rising,oscillation_rising, _) = oscillation_function(E_now,rising_sun_f[name])
        ax1.plot(interval_rising.T,oscillation_rising.T,linestyle='-',\
                 lw=2, label="$\omega^{\mathscr{R}_{\mu}}$: "+name,color='gray')

        ax1.title.set_text('Oscillation function of $\omega^{\mu}$ and $\omega^{\mathscr{R}_{\mu}}$')
        ax1.title.set_fontsize(28)
        ax1.legend(loc=4,prop={'size': 22})
        ax1.set_ylabel('$\omega$',size=28)
        ax1.grid(True)

        ax2 = plt.subplot(212)
        new_label = name+", noise:" +str(round(estimatE_now,4))
        ax2.plot(interval[name][0,50:200].T,oscillation[name][0,50:200].T,linestyle='-',lw=2,\
                 label="$\omega^{\mu}$: "+name,color='r')
        (interval_rising,oscillation_rising, _) = oscillation_function(E_now,rising_sun_f[name])
        ax2.plot(interval_rising[0,50:200].T,oscillation_rising[0,50:200].T,linestyle='-',\
                 lw=2, label="$\omega^{\mathscr{R}_{\mu}}$: "+name,color='gray')
        ax2.legend(loc=4,prop={'size': 22})
        ax2.grid(True)

    #figure_save("Oscillation denoising")
    plt.show()

    # 

    #print(np.sum(1*(oscillation_rising>oscillation[names[0]])))
    #print(np.sum(1*(oscillation_rising==oscillation[names[0]])))
    #print(np.sum(1*(oscillation_rising<oscillation[names[0]])))

    #

    from sklearn.linear_model import LinearRegression

    def decay_rate(
                   parameters,hyperparameters,name,omega_oscillation,\
                   jumps,vector_height_threshold,vector_peak_loc,\
                   distances,j
    ):
        '''
        Given a vector with length N, compute the decay either as average or as a exponential rate of decay
        decay_rate_type == min_max or 'regression'
        ----------------
        Input:
        -parameters: dictionary with materials' properties
        -hyperparameters: hyperparamters dictionary
        -name: name of material whose decay is being studied
        -omega_oscillation: oscillation function
        -jumps: jumps[i] gives the value |intensity_peak[i-1] - intensity_peak[i]|
        -vector_height_threshold: the vector with the series (h_*^{(0)}, h_*^{(1)}, ..., h_*^{(j)})
        -vector_peak_loc: vector with peak locations (oth peak, 1st peak,...., jth peak)
        -distances: distance between peaks, up to distance between peak[j-1] and peak[j]
        -j: last peak that we have found
        ----

        It depends on the decay_rate_type:
        -If decay_rate_type == min_max:
            - hyperparameters used: lambda_d and lambda_h
            ## See paper for further explanation

        -If decay_rate_type == reversed:
            Uses the min-max method in the reverse passage.
            - hyperparameters used:lambda_d and lambda_h
            ## See paper for further explanation

        -If decay_rate_type == learn_to_trust:
            Uses same method as min_max to estimate h_threshold, but uses weighted method to average distance
            - hyperparameters used: lambda_d and lambda_h

        -If decay_rate_type == regression:
            Uses regression in a weighted fashio to estimate distance/plateaus size (see paper, equation (8))

            For the height threshold, uses a regression to estimate an exponential fit to XANES curve 
            to the right of jth-peak, then uses covariance matrix to estimate amount of oscillation in the remaining 
            curve in the same interval

            - hyperparameters used: lambda_d,  lambda_h, and stretching_factor
        ----------------
        Output:
        ----------------
        -decay_factor: alpha_n in the paper
        - dist_estimated: estimated distance/plateaus size
        - height_threshold: estimated height threshold
        - decay_plot dictionary with information about regression (only used when decay_rate_type == regression)

        ----------------
        '''
        # Unpacking ....
        # ...parameters
        parameters_now = parameters[name]
        xanes_now = parameters_now['xanes']
        E_now =parameters_now['energy']
        stop = parameters_now['stop']    

        # ... hyperparameters
        lambda_d = hyperparameters['lambda_d']
        lambda_h = hyperparameters['lambda_h']
        decay_rate_type = hyperparameters['decay_rate_type']
        start = vector_peak_loc[j-1]+1 #peak_loc[str(j-1)+name]

        ###################################
        if (decay_rate_type =='min_max'):
            decay_factor = (np.max(xanes_now[0,start:]) -np.min(xanes_now[0,start:]))/omega_oscillation[0,-1]
            height_threshold = decay_factor*lambda_h*vector_height_threshold[j-1]

            #### Old way
            ##dist_estimated = int( max(decay_factor,lambda_d)*np.average(distances))
            dist_estimated = max(int( max(decay_factor,lambda_d)*np.average(distances)),2)

            return decay_factor, dist_estimated,height_threshold,None

        ###################################
        elif decay_rate_type =='reversed':

            decay_factor = (np.max(xanes_now[0,start:]) -np.min(xanes_now[0,start:]))/omega_oscillation[0,-1]
            height_threshold = decay_factor*lambda_h*vector_height_threshold[j-1]
            dist_estimated = max(int(lambda_d*np.average(distances)),2)

            return decay_factor, dist_estimated,height_threshold,None

        ###################################

        elif decay_rate_type == 'learn_to_trust':

            decay_factor = (np.max(xanes_now[0,start:]) -np.min(xanes_now[0,start:]))/omega_oscillation[0,-1]
            height_threshold = decay_factor*lambda_h*vector_height_threshold[j-1]
            ## Take into account that j is always greater than 1
            dist_estimated = max( int(((lambda_d*np.exp(-(j-1)) +(j-1)/2)/j)*2*np.average(distances)),2)

            return decay_factor, dist_estimated,height_threshold,None

        ###################################

        else: #decay_rate_type = 'regression':
            decay_plot={}
            ###################################
            # We begin with aa log-regression for distance
            lin_reg_dist = LinearRegression()
            interval = np.reshape(np.arange(j),(-1,1))
            lin_reg_dist.fit(interval,np.log(np.reshape(distances,(-1,1))) )
            predicted_distance = np.exp(lin_reg_dist.predict(np.reshape(interval[-1,0],(-1,1))))
            dist_estimated = max( int(((lambda_d*np.exp(-(j-1)) +(j-1)/2)/j)*2*predicted_distance),2)

            decay_plot['regression_for_distance'] =lin_reg_dist

            ###################################

            #Then we use the estimated distance to do a regression for the height threshold
            stretching_factor= hyperparameters['stretching_factor']
            lin_reg_jumps = LinearRegression()
            E_for_pred = np.reshape(E_now[0,start:start+int(2*dist_estimated)],(-1,1))
            xanes_for_pred = np.reshape(xanes_now[0,start:start+int(2*dist_estimated)],(-1,1)) 

            lin_reg_jumps.fit(E_for_pred,xanes_for_pred)

            # predicted values. 
            pred = lin_reg_jumps.predict(E_for_pred)
            v = xanes_for_pred - pred

            residual = np.power(np.linalg.norm(v),2)/(len(E_for_pred)-2)
            X = np.c_[np.ones([len(E_for_pred),1]),E_for_pred]

            ## Preparation steps to use the covariance matrix
            M = np.linalg.inv(np.matmul(np.transpose(X),X))

            error = np.sqrt(np.diag(np.matmul(np.matmul(X,M),X.T))*residual)
            error = np.reshape(error,(-1,1))
            decay_plot ={'range_energy_pred':E_for_pred,'range_xanes_pred':pred,'error':error}        

            jumps_average = np.average(jumps)

            min_est = min(
                xanes_for_pred[0,0]- stretching_factor*error[0,0],\
                xanes_for_pred[-1,0]- stretching_factor*error[-1,0] 
            )
            max_est= max(
                xanes_for_pred[0,0]+ stretching_factor*error[0,0],\
                xanes_for_pred[-1,0]+ stretching_factor*error[-1,0] 
            )
            estimated_jump = max_est- min_est
            ##########################################
            decay_factor = estimated_jump/jumps_average
            height_threshold = decay_factor*vector_height_threshold[0]#np.average(vector_height_threshold)

            return decay_factor, dist_estimated,height_threshold,decay_plot 

    #

    def estimate_distance(parameters,hyperparameters,name,rising_sun,oscillation):
        '''
        This function estimate the distance between local maxixanes_nowm and minixanes_nown in order to bootstrap the algorithm. 
        The first step consists in estimating x_1, which will then be e_1
        and posteriorly estimates x_0
        ----------------
        Input:
        - parameters: dictionary with materials' properties
        - hyperparameters: hyperparamters dictionary
        - name: name of material whose decay is being studied
        - rising_sun: the rising sun function associated with xanes_now
        - oscillation, which given the intrinsic oscillation of xanes_now
        ----------------
        Output:
        The function output are
        (x_0,x_1): the location of the minixanes_nowm and that of the maxixanes_nowm in the half-interval  
        ----------------
        '''
        # Unpacking
        xanes_now= parameters[name]['xanes']
        lambda_h_find_1st = hyperparameters['lambda_h_find_1st']
        lambda_d_find_1st = hyperparameters['lambda_d_find_1st']

        ## We want tthe raw xanes measurement, prior to the embedding, so we do the following
        m = xanes_now.shape[1];
        new_xanes_now = np.array(xanes_now[0,0:int(m/2)],ndmin=2)  

        '''
        In the next part we choose the smallest argument for which the spectrum assumes its maxixanes_nowm point
        #x_1 = min(np.where(np.max(new_xanes_now)==new_xanes_now)[1])
        A second version consists of finding the first peak with thresholds (4*oscillation, length/4)
        '''
        ################################################################  
        locations, heights = plateau_detection(rising_sun,int(lambda_d_find_1st*m+1))

        # Now we retain only the elements that are higher than the threshold
        what_matters= np.where(heights>=min(lambda_h_find_1st*oscillation,.5))    ### Lmabda_h ==5 before
        locations, heights = locations[what_matters], heights[what_matters]
        x_1 = locations[0]
        #################################################################

        min_xanes_now = max(np.min(new_xanes_now), rising_sun[0,0]) ## WE do this in case the function starts in a valley

        x_0 = np.copy(x_1)
        while( new_xanes_now[0,x_0]>=min_xanes_now + oscillation): 
            x_0 =x_0 - 1
        # obviously x_0 =< x_1

        return (x_0, x_1)

    #

    def find_peak(interval,f,h_threshold,L_threshold,num_iterations, iteration_flag,hyperparameters):
        '''
        This function finds a peak of the function f in an interval with certain height and spatial 
        thresholds. It does not have to estimate the tresholds

        ----------------
        Input:
        - interval: which is a row vector of shape 1xm
        - f: which is a row vector of shape 1xm
        - h_threshold: which denotes the peak threshold
        - L_threshold: an integer which denotes the plateau threshold
        -iteration_flag : only used in the case that no peak is found. Whenever that happens, it increase the 
                           iteration flag by 1. The upper bound for iteration flag is 20 
                            (hardcoded, but could be new hyperparameter)
        ----------------
        Output:
        - locations[0]:  index of peak location (integer)
        - interval[0,locations[0]]:  array value  of peak location (float)
        - heights[0]:  intensity at peak location (float)
        - num_iterations:  how many iterations were used.
        ----------------
        '''

        # Unpacking 
        iteration_decay = hyperparameters['iteration_decay']
        f_maximal = rising_sun(f)
        locations, heights = plateau_detection(f_maximal,L_threshold)

        # Now we detect the plateaus of f_maximal; the peak will be where the plateaus are located

        if (len(locations)==0): #locations==None):
            print("In find_peaks: L_threshold is", L_threshold,"iteration_flag", iteration_flag)
            if L_threshold >2 and (iteration_flag<=20):  
                if iteration_flag%2 ==0:
                    return find_peak(
                    interval,f,iteration_decay*h_threshold,L_threshold,num_iterations,iteration_flag+1,hyperparameters
                )

                else:
                    return find_peak(
                    interval,f,h_threshold,L_threshold-1,num_iterations,iteration_flag+1,hyperparameters
                )
                '''
           if L_threshold >2 and (iteration_flag<=10):       
                return find_peak(
                    interval,f,iteration_decay*h_threshold,L_threshold-1,num_iterations,iteration_flag+1,hyperparameters
                )
                '''
            else:
                return (None,None,None,None)

        #now we  just need to check whether heigths is bigger than the threshold
        # notice that we can always assume that if location has the 0 index, then heights is also 0. 
        #This is due to normalization

        # Now we retain only the elements that are higher than the threshold
        what_matters= np.where(heights>=h_threshold)
        locations, heights = locations[what_matters], heights[what_matters]

        if len(locations)==0: 
            print("In find_peaks: L_threshold is", L_threshold,"iteration_flag", iteration_flag)
            if L_threshold >2 and (iteration_flag<=20):  
                if iteration_flag%2 ==0:
                    return find_peak(
                    interval,f,iteration_decay*h_threshold,L_threshold,num_iterations,iteration_flag+1,hyperparameters
                )

                else:
                    return find_peak(
                    interval,f,h_threshold,L_threshold-1,num_iterations,iteration_flag+1,hyperparameters
                )
            else:
                return (None,None,None,None)

        else: return (locations[0], interval[0,locations[0]], heights[0],num_iterations)

    #

    def find_first_peak(
        parameters,hyperparameters, name,rising_sun,toscillation, oscill_threshold,first_peak_threshold
    ):

        '''
        This function finds the first peak of the function f in an interval with certain height and spatial 
        thresholds. Unlike "find_peak", it needs to estimate the thresholds

        ----------------
        Input:
        - parameters: dictionary with materials' properties
        - hyperparameters: hyperparamters dictionary
        - name: name of material whose decay is being studied
        - rising_sun
        , oscill_threshold: height_threshold, as given by Def I.4 in the paper
        - toscillation: oscillation function for current XANES measurement
        - first_peak_threshold: see explanation in previous markdown
        - rising_sun:  the Rising Sun function
        - L_threshold: an integer which denotes the plateau threshold
        -iteration_flag : only used in the case that no peak is found. Whenever that happens, it increase the 
                           iteration flag by 1. The upper bound for iteration flag is 20 
                            (hardcoded, but could be new hyperparameter)
        ----------------
        Output:

        - tpeak_loc0:  index of peak location (integer)
        - tpeak_energy0:  array value  of peak location (float)
        - tpeak_height0, :  intensity at peak location (float)
        - tnum_itera0:  how many iterations were used.
        - tdist_peak:  distance to previous peak
        - first_jump: jump in intensity (absolute value) when compared to previous peak

        ----------------
        '''
         # Unpacking
        lambda_d_shrink_1st = hyperparameters['lambda_d_shrink_1st']
        parameters_now = parameters[name]
        xanes_now = parameters_now['xanes']
        E_now =parameters_now['energy']

        ### Initial estimate for the distance between peaks, which we will use to feed the algorithm
        min_location, max_location = estimate_distance(parameters,hyperparameters,name,rising_sun,oscill_threshold)
        tdist_peak= np.array([int(lambda_d_shrink_1st*(max_location- min_location))],ndmin=1)  ## IT WAS 1/2 before

        # Find peak locations:
        iteration_flag=0
        tpeak_loc0,tpeak_energy0,tpeak_height0,tnum_itera0 = \
        find_peak(E_now,rising_sun,first_peak_threshold,tdist_peak[-1],0,iteration_flag,hyperparameters)

        # Last, now that we really know where the first peak is, we update it:
        tdist_peak=np.array([int(lambda_d_shrink_1st*(tpeak_loc0-min_location))]) ## IT WAS 1/2 before
        first_jump = xanes_now[0,tpeak_loc0] - xanes_now[0,min_location]

        return (tpeak_loc0, tpeak_energy0, tpeak_height0, tnum_itera0, tdist_peak, first_jump)

    #

    def print_brkpt_properties(j,peak_loc, peak_height, dist_peak, jumps):
        '''
        This function print the rpeakpoints properties:

        ----------------
        Input:
        - j: peak number
        - peak_loc: location of peak
        - peak_height: intensity of peak
        - dist_peak: distance to previous peak
        - jumps: intensity variation (in absolute value) with respect to previous peak
        ----------------
        Output:
        None
        ----------------
        '''
        print("\t Peak "+ str(j) +" located at: "+ str(peak_loc)) 
        print("\t Peak height:"+ str(peak_height) )    
        print("\t Distance between successive crests and peaks: ",dist_peak,"\n")
        print("\t Jumps: ",jumps,"\n")

    #

    def hidden_breakpoint(xanes_now,left,middle,right,j):
        '''
        Implements hidden peak trick
        ----------------
        Input:
        left<=middle<=right, three integer indexes
        j: j checks wheter we need to look for a hidden crest or a hidden valley
        ----------------
        Output:
        It depends on the input. 
            -If there is no need for it, it returns the midle.
            - If there is need for it, it will return the minimum  of the vector 

        ----------------
        '''   
        aux = xanes_now[0,left:right+1]
        if j%2==0:  M= min(aux)
        else: M= max(aux)

        if xanes_now[0,middle] ==M: ## In this case there is no need for the hidden peak trick
            return middle
        else: # in this case there was a problem, and we have to use the hidden peak trick
            return left+np.min(np.where(M == aux))

    #

    search_conditions = {
        'move':'backstep',#'middle_step',
        'printing':True,
        'polite_guess':{},
        'forward':True
    }

    #

    def rising_sun_envelope_method(
        parameters,hyperparameters, name, N_split_before,N_split_after,search_conditions
    ):
        '''
        ----------------
        Input:
        - parameters: dictionary with materials' properties
        - hyperparameters: hyperparamters dictionary
        - name: name of material whose decay is being studied
        - N_split_before:
        - N_split_after:
        - search_conditions: dictionary with search conditions:
            - 'move':'backstep' or 'middle_step',
            'printing': if middle computations should be printed
            'polite_guess':{}, this is empty in the forward passing, but it is filled with first peak 
                            information in the reverse passing
            'forward':True or False, denotes which passage of the algorithm we are, that is, whether
                        looking fo peaks on the left of 0th peak or on the right
        ----------------
        Output:
        A dictionary - decomposition-  with the following keys:


        - peak_loc: dictionary index of peak location (integer), counted from 0th peak (positive to right, negative to the left)
        - peak_energy: array value  of peak location (float) 
        - peak_height: intensity at peak location (float)
        - num_itera: how many iterations were used in inner loop that deals with peak not found
        - dist_peak: vector with  distance between scuccessive peaks
        - oscillation: oscillation function
        - height_threshold: vector with the series of height thresholds
        - jumps: jump in intensity (absolute value) when compared to previous peak
        - number_split_before_peak: number of splits after 0th peak (including it)
        - number_split_after_peak: number of splits after 0th peak (excluding it)
        - oscil_jump_ratio: at jth >0 (resp. j<0) entry, 
                            ratio between XANES curve from from peak j to end of measurement (begining to jth peak)
                            and the jump to previous (resp., next) peak
        ----------------
        '''
        ##############################################################
        # UNPACKING...
        # ... parameters
        parameters_now= parameters[name]
        stop = parameters_now['stop']    
        xanes_now = parameters_now['xanes']
        E_now =parameters_now['energy']

        # ...  hyperparameters
        lambda_h = hyperparameters['lambda_h']
        lambda_d = hyperparameters['lambda_d']
        init_oscil_par = hyperparameters['initial_oscillation_guess_parameter']

        # ...  search condition parameters
        move=search_conditions['move']
        printing=search_conditions['printing']
        polite_guess=search_conditions['polite_guess']
        forward = search_conditions['forward']
        ##############################################################

        if printing: 
            print("************************************** \n")
            print("Material:", name,"\n")

        start = 0

        # We shall save the resuls as dictionaries and vectors
        peak_loc, peak_energy, peak_height, num_itera,height_threshold = {},{},{},{},{}
        dist_peak,  jumps = [],[]

        ### Initializing some counters
        number_split_after_peak, number_split_before_peak, iteration_flag=0,0,0

        ## For book keeping purposes
        vector_peak_loc, vector_height_threshold =np.array([],dtype=np.int16), np.array([],dtype=np.float16)

        for j in range(max(N_split_after,1)):
            if (j==0):
                '''
                REMARK: j =0 is a case that is only considered in the forward passing.
                         It is necessary in order to find the first peak
                         This first part can be used as a bootstrap in order to find the distance:
                '''   
                if forward==False:
                    '''
                    If false, then we are in the reverse case, which means that these quantities need to be initialized
                    '''
                    # First of all, we recover the parameters we are going to use
                    peak_loc[str(0)] = polite_guess['peak_loc_initial']
                    rising_sun_f= polite_guess['initial_rising_sunf_f']
                    height_threshold[str(0)]= polite_guess['initial_height_threshold']
                    oscillation = polite_guess['initial_oscillation']
                    critical_threshold= polite_guess['initial_critical_threshold']
                    dist_peak = polite_guess['dist_peak_name']
                    jumps = np.array([0])

                    ### Book keeping
                    vector_peak_loc = np.append(vector_peak_loc,peak_loc[str(0)])
                    vector_height_threshold = np.append(vector_height_threshold,height_threshold[str(0)])
                    continue

                if printing: 
                    print("************************************** \n")
                    print("\t Going for " +str(j)+"th breakpoint. \n")

                rising_sun_f = rising_sun(xanes_now)        

                ### This is where the oscillation function gets calculated
                _, oscillation, critical_threshold= oscillation_function(E_now,xanes_now)

                #This is where the compute the height threshold
                height_threshold["0"]=(int(oscillation[0,-1]/(init_oscil_par*critical_threshold)) + 1)*critical_threshold

                '''And this is where we look for the first peak
                   We remark that it is here that part of the hyperparameters are used, in the function
                   find_fisrst_peak > estimate_distance'''
                peak_loc["0"], peak_energy["0"], peak_height["0"],\
                num_itera["0"], dist_peak, jumps=\
                find_first_peak(
                    parameters,hyperparameters, name,\
                    rising_sun_f,oscillation, critical_threshold,\
                    height_threshold["0"]
                )

                ### Book keeping
                vector_peak_loc = np.append(vector_peak_loc,peak_loc[str(j)])
                vector_height_threshold = np.append(vector_height_threshold,height_threshold[str(j)])

                if printing:  print_brkpt_properties(j,peak_loc[str(j)], peak_height[str(j)], dist_peak, jumps)

                number_split_after_peak +=1    

            #secondary cases - for which we have to use a maximal function
            elif j%2==0:
                if printing: 
                    print("\t ***************************** \n")
                    print("\t Going for " +str(j)+"th breakpoint. \n")
                start= peak_loc[str(j-1)]+1
                interval = np.array(E_now[0,start:],ndmin=2)

                #Truncate to the appropriate interval
                height_normalizer =  np.copy(xanes_now[0,start])
                f = np.array(xanes_now[0,start:],ndmin=2) -height_normalizer       ## you always normalize
                rising_sun_f = rising_sun(f)

                # Update the definition of the height threshold
                decay_factor, temp,height_threshold[str(j)],_ =\
                decay_rate(
                    parameters,hyperparameters,name,oscillation, jumps,\
                    vector_height_threshold,vector_peak_loc, dist_peak,j
                )
                tpeak_loc, tpeak_energy, tpeak_height, tnum_itera=\
                find_peak(
                    interval,f,height_threshold[str(j)],temp,0,iteration_flag,hyperparameters
                )           

                if (tpeak_loc ==None) or (tpeak_loc+ start >=stop):  ### Then you can stop here!!!
                    if printing: 
                        print("\n ONLY "+ str(number_split_after_peak)+" SPLITTINGS WERE POSSIBLE!\n") 
                    break   

                peak_loc[str(j)], peak_energy[str(j)],peak_height[str(j)],num_itera[str(j)]=\
                tpeak_loc,tpeak_energy,tpeak_height,tnum_itera

                # We go back to the non-truncated vector
                peak_loc[str(j)] = peak_loc[str(j)] +start

                ### Book keeping
                vector_peak_loc = np.append(vector_peak_loc,peak_loc[str(j)])
                vector_height_threshold = np.append(vector_height_threshold,height_threshold[str(j)])

                ##Now we need to check whether hidden peak trick is necessary
                if j>=2:
                    p = hidden_breakpoint(xanes_now,peak_loc[str(j-2)],peak_loc[str(j-1)],peak_loc[str(j)],j)
                    ## MIDDLE MOVE
                    if (move== 'middle_step') or (peak_loc[str(j-1)]==p):
                        peak_loc[str(j-1)]=p
                        peak_energy[str(j-1)]=E_now[0,p]
                        peak_height[str(j-1)]=xanes_now[0,p]
                        dist_peak[-1] = int(peak_loc[str(j-1)]- peak_loc[str(j-2)])

                        ### Recompute these quantities
                        start= peak_loc[str(j-1)]+1
                        #Truncate to the appropriate interval
                        height_normalizer =  np.copy(xanes_now[0,start])
                        f = np.array(xanes_now[0,start:],ndmin=2) -height_normalizer       ## you always normalize
                        rising_sun_f = rising_sun(f)

                        # Recalculate decay factor
                        vector_peak_loc[j-1] =peak_loc[str(j-1)]
                        decay_factor, _,height_threshold[str(j)],_ =\
                        decay_rate(
                            parameters,hyperparameters,name,oscillation, jumps,vector_height_threshold,vector_peak_loc, dist_peak,j
                        )
                        jumps[-1] =  np.abs(xanes_now[0,peak_loc[str(j-1)]] - xanes_now[0,peak_loc[str(j-2)]])

                    else: ## BACKSTEP MOVE
                        ## Can be iterated, but we won't go for it
                        p = hidden_breakpoint(xanes_now,peak_loc[str(j-1)],p,p,j-1)

                        peak_loc[str(j)]=p
                        ### we don't need to recompute the quantities f and rising_sun_f, just the jump and others
                        peak_energy[str(j)]=E_now[0,p]

                #############################################################           
                #upgrade the dist_peak matrix

                dist_peak = \
                np.append(dist_peak,np.array([int(peak_loc[str(j)]- peak_loc[str(j-1)])]) ) 

                # unnormalize peak height
                peak_height[str(j)] = xanes_now[0,peak_loc[str(j)]]#==peak_height[str(j)]+height_normalizer

                # Book keeping    
                jumps = np.append(jumps, np.abs(xanes_now[0,peak_loc[str(j)]] - xanes_now[0,peak_loc[str(j-1)]]))

                if printing: 
                    print_brkpt_properties(j,peak_loc[str(j)], peak_height[str(j)], dist_peak, jumps)

                number_split_after_peak +=1    

            ## third case, for which we use the valley_of_shadows function

            elif j%2 ==1:   
                if printing: 
                    print("\t ***************************** \n")
                    print("\t Going for " +str(j)+"th breakpoint. \n")
                # First: set up the interval of relevance
                start= peak_loc[str(j-1)]+1
                interval = np.array(E_now[0,start:],ndmin=2)

                #At this point we do the following: 
                #1) We truncate the maximal function from start to end
                #2) Construct first difference
                rel_start = start
                if j>1: rel_start -= (peak_loc[str(j-2)]+1)
                rising_sun_f = np.array(rising_sun_f[0,rel_start:],ndmin=2)
                f = np.array(xanes_now[0,start:],ndmin=2)
                valley_of_shadows = rising_sun_f- f #this function will always be nonnegative, thanks to (1)  

                ## Now we renormalize, because it has to start from 0
                valley_of_shadows = valley_of_shadows- valley_of_shadows[0,0] 
                f = np.array(valley_of_shadows,ndmin=2)

                ## UPDATES...
                #... of height threshold
                # Update the definition of the height threshold
                decay_factor, temp,height_threshold[str(j)],_ =\
                decay_rate(
                    parameters,hyperparameters,name,oscillation, jumps,\
                    vector_height_threshold,vector_peak_loc, dist_peak,j
                )

                tpeak_loc, tpeak_energy, tpeak_height, tnum_itera=\
                find_peak(interval,f,height_threshold[str(j)],temp,0,iteration_flag,hyperparameters)           

                if (tpeak_loc ==None) or (tpeak_loc+ start >=stop):  ### Then you can stop here!!!
                    if printing: 
                        print("\n ONLY "+ str(number_split_after_peak)+" SPLITTINGS WERE POSSIBLE!\n") 
                    break   

                peak_loc[str(j)], peak_energy[str(j)],peak_height[str(j)],num_itera[str(j)]=\
                tpeak_loc,tpeak_energy,tpeak_height,tnum_itera

                # We go back to the non-truncated vector
                peak_loc[str(j)] = peak_loc[str(j)]+ start

                ### Book keeping
                vector_peak_loc = np.append(vector_peak_loc,peak_loc[str(j)])
                vector_height_threshold = np.append(vector_height_threshold,height_threshold[str(j)])

                ##############################################################
                ##Now we need to check whether hidden peak trick is necessary

                if j>=2:
                    p = hidden_breakpoint(xanes_now,peak_loc[str(j-2)],peak_loc[str(j-1)],peak_loc[str(j)],j)

                    if (move== 'middle_step') or (peak_loc[str(j-1)]==p):
                        peak_loc[str(j-1)]=p
                        peak_energy[str(j-1)]=E_now[0,p]
                        peak_height[str(j-1)]=xanes_now[0,p]
                        dist_peak[-1] = int(peak_loc[str(j-1)]- peak_loc[str(j-2)])

                        ### Recompute these quantities
                        start= peak_loc[str(j-1)]+1
                        #Truncate to the appropriate interval
                        height_normalizer =  np.copy(xanes_now[0,start])
                        f = np.array(xanes_now[0,start:],ndmin=2) -height_normalizer       ## you always normalize
                        rising_sun_f = rising_sun(f)

                        # Recalculate decay factor
                        vector_peak_loc[j-1] =peak_loc[str(j-1)]
                        #vector_height_threshold[j]= height_threshold[str(j)] 
                        decay_factor,_,height_threshold[str(j)],_ =\
                        decay_rate(
                            parameters,hyperparameters,name,oscillation, jumps,vector_height_threshold,vector_peak_loc, dist_peak,j
                        )
                        jumps[-1] =  np.abs(xanes_now[0,peak_loc[str(j-1)]] - xanes_now[0,peak_loc[str(j-2)]])

                    else: ## BACKSTEP MOVE
                        ## Can be iterated, but we will iterate only once
                        p = hidden_breakpoint(xanes_now,peak_loc[str(j-1)],p,p,j-1)

                        peak_loc[str(j)]=p
                        ### we don't need to recompute the quantities f and rising_sun_f, just the jump and others
                        peak_energy[str(j)]=E_now[0,p]

                ##############################################################
                #upgrade the dist_peak matrix
                dist_peak = \
                np.append(dist_peak,np.array([int((peak_loc[str(j)]- peak_loc[str(j-1)]))]) )        

                peak_height[str(j)] =xanes_now[0,peak_loc[str(j)]]        
                jumps = np.append(jumps, np.abs(xanes_now[0,peak_loc[str(j)]] - xanes_now[0,peak_loc[str(j-1)]]))

                if printing: 
                    print_brkpt_properties(j,peak_loc[str(j)], peak_height[str(j)], dist_peak, jumps)

                number_split_after_peak +=1

        # In this part we figure out whether the algorithm goes for a second round of measurements, but now using 
        # the previous estimates on distance to improve the accuracy of this estimate

        ## Compute oscillation metric:
        oscil_jump_ratio ={} ### 
        x, y = np.array(E_now[0,peak_loc["0"]:stop],ndmin=2),np.array(xanes_now[0,peak_loc["0"]:stop],ndmin=2)
        _,_, amount_oscil = oscillation_function(x,y, full_computation=False)
        #oscil_jump_ratio["0"] =  amount_oscil/jumps[0]
        
        epsilon = 10**(-5) # Adição de um termo infinitesimal para evitar divisão por 0 (zero). (Será que adicionar um infinitesimal foi ruim??)
        oscil_jump_ratio["0"] =  amount_oscil/(jumps[0]+epsilon)

        

        ##################
        # We need to do this because the algorithm is asymetric
        flag_oscillation =1
        if forward:
            flag_oscillation =0

        for i in range(1,number_split_after_peak +flag_oscillation):
            x, y = np.array(E_now[0,peak_loc[str(i)]:],ndmin=2),np.array(xanes_now[0,peak_loc[str(i)]:],ndmin=2)
            _,_, amount_oscil = oscillation_function(x,y, full_computation=False)
            oscil_jump_ratio[str(i)] =  amount_oscil/jumps[i]
            if np.isnan(oscil_jump_ratio[str(i)]):
                oscil_jump_ratio[str(i)] = np.inf
        ##################


        if N_split_before==0:
            if printing: 
                print("\n ONLY "+ str(number_split_after_peak+number_split_before_peak)+" SPLITTINGS \
                TO THE RIGHT OF THE MAIN PEAK WERE POSSIBLE!") 

            decomposition={
                'peak_location':peak_loc,'peak_energy':peak_energy,
                'peak_heights':peak_height,'number_iterations':num_itera,
                'distance_between_peaks':dist_peak, 'oscillation':oscillation,
                'height_threshold_evolution':height_threshold,'jumps':jumps,
                'number_of_splittings_before':number_split_before_peak, 
                'number_of_splittings_after':number_split_after_peak,
                'oscil_jump_ratio':oscil_jump_ratio
            }

            #print(f'Esse é o valor de peak_location do {filename}: {decomposition["peak_location"]}')
            #print(f'Esse é o valor de peak_heights do {filename}: {decomposition["peak_heights"]}')
            #print(f'Esse é o valor de peak_energy do {filename}: {decomposition["peak_energy"]}')
            print(f'Esse é o dicionário do {filename}: {decomposition}')
            print(f'A quantidade de picos identificados nesse espectro é: {len(decomposition["peak_location"])}')
            print("Razão de oscilação_1:", decomposition['oscil_jump_ratio'])
        
            return decomposition
            

        ###############################################################################################      
        ### REVERSE PASSING:
        ### we reverse the spectra to do the search on the other side of the mountain
        ### At this point, the algorithm is using the previous data to improve the location of the peaks    
        ###############################################################################################      

        if printing: 
            print("************************************** \n")
            print("\t STARTING REVERSE PASSING, ON THE LEFT SIDE\n")
        # We want to include the peak_loc(0)
        chopped_xanes = np.array(xanes_now[0,:peak_loc["0"]+1],ndmin=2)
        chopped_E = np.array(E_now[0,:peak_loc["0"]+1],ndmin=2)

        ## Now we reverse these vectors
        chopped_xanes = np.array(chopped_xanes[0,::-1],ndmin=2)
        chopped_E = np.array(chopped_E[0,::-1],ndmin=2)

        ### REMARK: THE ABOVE TWO OPERATIONS ARE NOT COMMUTATIVE!!!

        if number_split_after_peak >1: # Then we can remove the first element
            dist_peak = dist_peak[1:]  # REMOVE FIRST ELEMENT!!!
            jumps = jumps[1:]  # REMOVE FIRST ELEMENT!!!

        ## And then we pad these vectors
        reverse_E,reverse_xanes, reverse_stop = \
        small_padded_spectra(chopped_E,chopped_xanes)

        # Things that we are going to use in the REVERSE PASSING
        # ... parameters
        reverse_N_split_after= N_split_before+1  # add 1 because we are not looking for 0 anymore
        reverse_N_split_before=0

        ##################################################################
        #Let's create a new parameters dictionary"
        reverse_parameters ={
            name:{'name':name,
                  'xanes':reverse_xanes,
                  'energy':reverse_E,
                  'stop':reverse_stop }
        }
        "...and a new hyperparameters dictionary"
        reverse_hyperparameters={
             ## These hyperparameters will be used in 
            'lambda_h':1,'lambda_d':1/4,
            ## These hyperparameters will be used in find_first_peak
            'lambda_h_find_1st':4,'lambda_d_find_1st':1/4,
            #'lambda_d_shrink_1st':1/2, # WON'T BE NEEDED
            'initial_oscillation_guess_parameter':10,
          ## These hyperparameters will be used in 
            'stretching_factor':3,'iteration_decay':.9,'decay_rate_type': 'reversed'
        }
        # ...  search condition parameters. But before we do that we need to update a few things
        reverse_polite_guess={
            'peak_loc_initial':int(0),
            'initial_rising_sunf_f': rising_sun(reverse_xanes),
            'initial_height_threshold': height_threshold[str(0)],
            'initial_oscillation':oscillation,
            'initial_critical_threshold':critical_threshold,
            'dist_peak_name': int(np.average(dist_peak/4)),
            'jump': np.average(jumps/4)
        }

        reverse_search_conditions = {
            'move':'middle_step','printing':False,
            'polite_guess':reverse_polite_guess,'forward':False
        }
        ##################################################################

        # Then we call the same algorithm on this reversed spectrum.

        reverse_decomposition= \
        rising_sun_envelope_method(
            reverse_parameters,reverse_hyperparameters,name,\
            reverse_N_split_before, reverse_N_split_after,reverse_search_conditions
        )

        # Now we unpack the result
        reverse_peak_loc = reverse_decomposition['peak_location']  
        reverse_peak_energy = reverse_decomposition['peak_energy'] 
        reverse_peak_height = reverse_decomposition['peak_heights'] 
        reverse_num_itera = reverse_decomposition['number_iterations']  
        reverse_dist_peak = reverse_decomposition['distance_between_peaks']   
        reverse_oscillation = reverse_decomposition['oscillation'] 
        reverse_height_threshold = reverse_decomposition['height_threshold_evolution'] 
        reverse_number_splittings_before = reverse_decomposition['number_of_splittings_before']
        reverse_number_splittings_after = reverse_decomposition['number_of_splittings_after']
        reverse_jumps = reverse_decomposition['jumps'] 
        reverse_oscil_jump_ratio = reverse_decomposition['oscil_jump_ratio'] 

        #And in the end we need to put things together.
        # The terms don't need reflection, for they are invariant

        if number_split_after_peak >1: # Then we can remove the first element
            reverse_dist_peak = reverse_dist_peak[1:]
            reverse_dist_peak = reverse_dist_peak[::-1]
            dist_peak = np.append(reverse_dist_peak,dist_peak)                  

            reverse_jumps = reverse_jumps[1:]
            reverse_jumps = reverse_jumps[::-1]
            jumps = np.append(reverse_jumps,jumps)
        else:
            reverse_jumps = reverse_jumps[1:]
            jumps = reverse_jumps[::-1]

            reverse_dist_peak = reverse_dist_peak[1:]
            dist_peak = reverse_dist_peak[::-1]

        number_split_before_peak = reverse_number_splittings_after
        for k in range(1,number_split_before_peak+1):
            # unnormalize peak height
            peak_loc[str(-k)] =peak_loc["0"] - (reverse_peak_loc[str(k)])
            peak_energy[str(-k)] = reverse_peak_energy[str(k)]
            peak_height[str(-k)] = reverse_peak_height[str(k)]
            num_itera[str(-k)]  = reverse_num_itera[str(k)]  
            height_threshold[str(-k)] = reverse_height_threshold[str(k)]   
            oscil_jump_ratio[str(-k)] = reverse_oscil_jump_ratio[str(k)]

        if printing: 
            print("\n ONLY "+ str(number_split_before_peak)+" SPLITTINGS \
                TO THE LEFT OF THE MAIN PEAK WERE POSSIBLE!\n\n") 
            print("\n In TOTAL, ONLY "+ str(number_split_after_peak+number_split_before_peak)+" SPLITTINGS WERE POSSIBLE!\n") 

        decomposition={
            'peak_location':peak_loc,'peak_energy':peak_energy,
            'peak_heights':peak_height,'number_iterations':num_itera,
            'distance_between_peaks':dist_peak, 'oscillation':oscillation,
            'height_threshold_evolution':height_threshold,'jumps':jumps,
            'number_of_splittings_before':number_split_before_peak, 
            'number_of_splittings_after':number_split_after_peak,
            'oscil_jump_ratio':oscil_jump_ratio
        } 
        return decomposition    

    #

    search_conditions['printing'] = False
    hyperparameters['lambda_d']=1/4
    hyperparameters['decay_rate_type']= 'min_max'#'learn_to_trust'

    #

    def write_material_peak_properties(
        parameters,hyperparameters,search_conditions,\
        names, N_split_after, N_split_before
    ):
        '''
           ----------------
        Input:
        - parameters: dictionary with materials' properties
        - hyperparameters: hyperparamters dictionary
        - names: names of materials studied
        - N_split_before: upper bound on number of peaks to the right (always set to be greater than 1,
                          because 0th peak is included) 
        - N_split_after:upper bound on number of peaks to the left 
        - search_conditions: dictionary with search conditions:
            - 'move':'backstep' or 'middle_step',
            'printing': if middle computations should be printed
            'polite_guess':{}, this is empty in the forward passing, but it is filled with first peak 
                            information in the reverse passing
            'forward':True or False, denotes which passage of the algorithm we are, that is, whether
                        looking fo peaks on the left of 0th peak or on the right
        ----------------
        Output:
        - decompositions is a dictionary with names elements as keys, and elements are decomposition dictionary. 
        Each  decomposition dictionary has the  following keys:

        - peak_loc: dictionary index of peak location (integer), counted from 0th peak (positive to right, negative to the left)
        - peak_energy:array value  of peak location (float)
        - peak_height: intensity at peak location (float)
        - num_itera: how many iterations were used in inner loop that deals with peak not found
        - dist_peak: vector with  distance between scuccessive peaks
        - oscillation: oscillation function
        - height_threshold: vector with the series of height thresholds
        - jumps: jump in intensity (absolute value) when compared to previous peak
        - number_split_before_peak: number of splits after 0th peak (including it)
        - number_split_after_peak: number of splits after 0th peak (excluding it)
        - oscil_jump_ratio: at jth >0 (resp. j<0) entry, 
                            ratio between XANES curve from from peak j to end of measurement (begining to jth peak)
                            and the jump to previous (resp., next) peak
        '''
        decompositions={}
        ## We are going to split in N_split features
        for name in names:
            decompositions[name]=\
            rising_sun_envelope_method(
                parameters,hyperparameters,name, N_split_before, N_split_after,search_conditions
            )

        return decompositions

    #

    N_split_after, N_split_before=30, 5

    decompositions =\
    write_material_peak_properties(
        parameters,hyperparameters,search_conditions,names,N_split_after , N_split_before
    )

    #

    name = names[0]
    decompositions[name].keys()

    #

    for name in names:
        print(
            name,"has",decompositions[name]['number_of_splittings_before']," splittings before, and",\
            decompositions[name]['number_of_splittings_after'],"splittings after\n"
        )

    # 

    def plot_marked_peak_spectrum(
        parameters,names,decompositions,\
        save_as,plot_type='normalized',\
        figure_extension="eps",loc=4,title = "XANES spectrum with marked peaks"
    ):
        '''
        This function plots the spectra in the dictionary parameters that are indexed by the elements in names, 
        and mark them with the peaks given in the dictionary decompositions
        The plots are saved with name save_as.
        Inputs can be normalized or not ('raw' case)

        ----------------
        Input:
        - parameters: dictionary with materials' properties
        - names: names of materials studied
        - decomposition: dictionary with information about peaks and their locations
        - plot_type: 'normalized' or 'raw'
        - figure_extension: 'png' or 'eps'
        - loc: number from {1,2, 3, 4,} where legend will be placed
        - title: title in the plot
        ----------------
        '''
        plt.figure(figsize=(15,10))
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)

        i = 0
        for name in names:
        ##############################################################
        # UNPACKING...
        # ... parameters
            # Unpacking
            stop = parameters[name]['stop']    
            if plot_type =='normalized':
                xanes_now = parameters[name]['xanes']
            elif plot_type =='raw':
                xanes_now = parameters[name]['raw_xanes']
            E_now =parameters[name]['energy']
            decomposition_now= decompositions[name]
            number_splittings_before = decomposition_now['number_of_splittings_before']
            number_splittings_after = decomposition_now['number_of_splittings_after']
            N = number_splittings_before+number_splittings_after

            plt.plot(
                E_now[0,0:stop].T,xanes_now[0,0:stop].T,\
                label =name+",#Breakpoints: "+str(N),color='C'+str(i%10),lw=3,linestyle='-',alpha=.7
            )
            plt.grid(True)
            i+=1

            # Plot a marker
            colors = ['g','y','r']

            proportion_noise_jump = decomposition_now['oscil_jump_ratio']
            values = [float(x) for x in list(proportion_noise_jump.values())]
            for i in range(len(values)):
                if np.isinf(values[i]):
                    values[i] = 3.0
                    
            peaks_temp_list = list() #JAMES EDIT - Aqui os .csv finais (10 primeiros pontos identificados, energia e intensidade de absor��o) ser�o criados!
            i = 0 #JAMES EDIT
            for plots_iter in range(-number_splittings_before,number_splittings_after):
               # First peak marker
                print('-------------- LOOP NUMBER',i,'----------------')
                position = decomposition_now['peak_location'][str(plots_iter)]
                plt.plot(
                    E_now[0,position],xanes_now[0,position],\
                    color=colors[min(int(np.floor(values[plots_iter])),2)], \
                    label=None,marker='H',lw=4, markersize=12,alpha=.8
                )
                #peaks_temp_list.append([i, E_now[0,position], xanes_now[0,position], min(int(np.floor(values[plots_iter])),2)]) #JAMES EDIT
                peaks_temp_list.append([E_now[0,position], xanes_now[0,position]]) #, decomposition_now['oscil_jump_ratio'][str(i)]]) #EMELYN EDIT
                print(peaks_temp_list)
                i += 1
            if i>8:
                new_temp_list = list()
                for j in range(0,10):
                    new_temp_list.insert(len(new_temp_list), peaks_temp_list[j][0])
                    new_temp_list.insert(len(new_temp_list), peaks_temp_list[j][1])
                    #new_temp_list.insert(-1, peaks_temp_list[j][2])
                print(new_temp_list)
                peaks_df = pd.DataFrame([new_temp_list],
                                        columns=['E0','mu0', #'q0',
                                                 'E1','mu1', #'q1',
                                                 'E2','mu2', #'q2',
                                                 'E3','mu3', #'q3',
                                                 'E4','mu4', #'q4',
                                                 'E5','mu5', #'q5',
                                                 'E6','mu6', #'q6',
                                                 'E7','mu7', #'q7',
                                                 'E8','mu8', #'q8',
                                                 'E9','mu9', #'q9',
                                                ])
                save_peaks=f"{filename}.csv" #JAMES EDIT
                peaks_df.to_csv(save_peaks)
                print('----------------------------------')
                print('SAVING PEAKS DATA TO', save_peaks)
                print('----------------------------------')

        #print("Antes do loop:", peaks_temp_list) #EMELYN EDIT
        print('----------------------------------')
        print("Depois do loop:", new_temp_list)
        print('----------------------------------')
        print("Razão de oscilação_2:", decomposition_now['oscil_jump_ratio'])
        print('----------------------------------')

            
        plt.title(title,size=28)
        plt.legend(loc=loc,prop={'size': 22})
        plt.xlabel('Energy (eV)',size=28)
        plt.ylabel('Absorption $\mu(E)$',size=28)
        figure_save(save_as,  figure_extension='png') 
        #plt.show()    

    #
    #print(f'RAZÃO DE OSCILAÇÃO: {decomposition_now['oscil_jump_ratio']}') #EMELYN EDIT 

    plot_marked_peak_spectrum(
        parameters,names,decompositions,\
        #save_as="XANES_with_marked_peaks_average-middle_step",\
        save_as=f"{filename}XANES_with_marked_peaks_average-middle_step",\
    )

    #

    plot_marked_peak_spectrum(
        parameters,names,decompositions,\
        #save_as="XANES_with_marked_peaks_average-middle_step",\
        save_as=f"{filename}XANES_with_marked_peaks_average-middle_step",\
        plot_type='normalized',
        figure_extension="png"
    )

    #

    plot_marked_peak_spectrum(
        parameters,{names[0]},decompositions,f"{filename}raw_XANES_with_marked_peaks_average",plot_type="raw"
    )
    
    #print(f'Esses são as localizações do picos do {filename}: {decomposition['peak_loc']}')
    #print(f'Esse é o `decompositions` do {filename}: {decompositions}')

    # FIM! :) 

