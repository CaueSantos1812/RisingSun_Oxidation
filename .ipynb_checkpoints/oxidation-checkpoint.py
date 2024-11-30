import CZDS_utils
import os
import regex as re
import glob
import shutil
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from CZDS_utils import read


def find_element_edge(file):
	'''
	função para encontrar as infomações de elementos e borda 
	dentro do cabeçalho do arquivo .xdi
	
	input: arquivo .xdi
	return: (simbolo do elemento, borda analisada), tupla
	'''
	
	# Define as seções de interesse
	sections = ["Element.symbol", "Element.edge"]
	regex = '|'.join(map(re.escape, sections))

	# Dicionário para armazenar valores extraídos
	element_info = {}

	lines = file.read()
	matches = re.findall(f'({regex}):\\s*(.*)', lines)

	for match in matches:
		section, value = match
		element_info[section] = value

	# Retorna apenas os valores para "Element.symbol" e "Element.edge"
	return element_info.get("Element.symbol"), element_info.get("Element.edge")

def find_foils(pasta_mae, arquivos):
	'''
	Função utilizada para encontrar os foils dentro da Cruzeiro do Sul,
	e move-los para uma pasta separada, criando a pasta se inexistente.
	Mas pode ser adaptada para encontrar outras informações dentro do
	nome dos arquivos da base de dados.
	
	inputs: nome da pasta mãe (tanto para criar, quanto para encontrar) e os arquivos (utlizando o .glob)
	return: none, mas move os arquivos com a informação desejada no nome para a pasta indicada
	'''
	# Pasta de destino para arquivos "foil"
	pasta_filha = os.path.join(pasta_mae, "xdi_Ferros")

	# Cria a pasta de destino, se não existir
	os.makedirs(pasta_filha, exist_ok=True)


	for filepath in arquivos:
		filename = os.path.basename(filepath)

		# Verifica se o nome do arquivo contém "foil", pode ser adaptado para verificar outra string
		if "foil" in filename.lower():

			# Move o arquivo para a pasta de referência de foils
			shutil.move(filepath, os.path.join(pasta_filha, filename))
			print(f"Movido: {filename} para {pasta_filha}")

		else:
			pass
			print(f"Ignorado: {filename} (não contém 'foil')")
			
			
			
def norm_e_mu(xdi):
    data = CZDS_utils.read.xdi(xdi)
    if data[0] == 'Transmission':
        if CZDS_utils.XASNormalization.xas_type(data[1]) == "EXAFS":
            enorm, mu_norm = CZDS_utils.XASNormalization.EXAFS_normalization(data[1], data[4], debug=False)
            return enorm, mu_norm
        else:
            enorm, mu_norm = CZDS_utils.XASNormalization.XANES_normalization(data[1], data[4], debug=False)
            return enorm, mu_norm

    elif data[0] == 'Transmission Raw':
        if CZDS_utils.XASNormalization.xas_type(data[1]) == "EXAFS":
           #tranmission_raw_exafs_counter += 1
            enorm, mu_norm = CZDS_utils.XASNormalization.EXAFS_normalization(data[1], data[2], debug=False)
            return enorm, mu_norm
        else:
            enorm, mu_norm = CZDS_utils.XASNormalization.XANES_normalization(data[1], data[2], debug=False)
            return enorm, mu_norm

    elif data[0] == 'Normalized Transmission':
        if czds_utils.XASNormalization.xas_type(data[1]) == "EXAFS":
            enorm, mu_norm = CZDS_utils.XASNormalization.EXAFS_normalization(data[1], data[2], debug=False)
            return enorm, mu_norm
        else:
            enorm, mu_norm = CZDS_utils.XASNormalization.XANES_normalization(data[1], data[2], debug=False)
            return enorm, mu_norm
    elif data[0] == 'Fluorescence':
       # fluorescence_counter += 1
        if CZDS_utils.XASNormalization.xas_type(data[1]) == "EXAFS":
            enorm, mu_norm = CZDS_utils.XASNormalization.EXAFS_normalization(data[1], data[2], debug=False)
            return enorm, mu_norm
        else:
            enorm, mu_norm = CZDS_utils.XASNormalization.XANES_normalization(data[1], data[2], debug=False)
            return enorm, mu_norm
		
		
        
def nfind(x, xmin):
    """
	Encontra o índice do elemento x mais próximo do x mínimo
	
	inputs: x (lista), xmin (valor float)
	return: índice do elemento x mais próximo do x mínimo
    """
    return np.argmin(np.abs(x - xmin))

def integrate_energy_mu(E, mu, eo, mini=-5., fini=7., int_lim_inf=-10, int_lim_sup=30, dE=0.1):
	
	'''
	função que implementa o método da QUATI, inspirado no método de Capehart para determinar estado de oxidação
	inputs: 
		E (lista): Energias analisadas,
		mu (lista): Valores de absorção),
		
		eo (float): Referência para o cálculo, representa valor de energia para qual a derivada é máxima na região de             pré-pico
		esse valor é tabelado, e pode ser encontrado em:
		<https://xraydb.xrayabsorption.org>,
		
		mini (float): valor que define o começo da região de pré-pico para ser excluída (float),
		fini (float): valor que define o final da região de pré-pico para ser excluída (float),
		int_lim_inf (float): limite inferior da integral,
		int_lim_sup (float): limite superior da integral,
		dE (float): Valor do "passo" da integral numérica,
	'''

	# Desloca as energias por e0
	DE = E - eo

	# Exclui a região de pré-pico definida
	args = np.where((DE > mini) & (DE < fini))
	En = np.delete(DE, args)
	mun = np.delete(mu, args)

	# Interpola na região excluída
	f2 = interp1d(En, mun, kind='cubic')
	Enew = np.arange(DE[1], DE[-5], dE)
	mu_new = f2(Enew)

	# Define os limites de integração
	n_i = nfind(Enew, int_lim_inf)
	n_f = nfind(Enew, int_lim_sup)
	Enew = Enew[n_i:n_f]
	mu_new = mu_new[n_i:n_f]

	# Realiza a integração acumulativa
	area = [0.]
	for i, dmu in enumerate(mu_new[1:], start=1):
		area.append(area[-1] + dmu * dE)

	return Enew, area

def delta_energy(Enew, area):
    """
    Parâmetros:
	Enew (array): Array de valores de energia dentro do intervalo de integração.
	area (array): Valores da integral cumulativa correspondentes a `Enew`.

	Retorna:
	float: Valor de energia interpolado para a 'area' fornecida.
    """
    # Create an interpolator for area vs. Enew
    Aux = interp1d(area, Enew, kind='cubic')
    return Aux(area[-1]/2)
