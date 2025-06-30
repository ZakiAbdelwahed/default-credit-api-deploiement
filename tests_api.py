import pytest
import requests
import logging

# Configuration du logging pour pytest
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

########################################## 1. Test fichier ne contenant que l'entête ##########################################

def test_fichier_entete_seul():
    """Test fichier ne contenant que l'entête"""
    # Définir l'url, le nom du test et le lien du fichier à utiliser
    url = "http://localhost:8000/predict"
    test_name = "Test fichier ne contenant que l'entête"
    file_path = "fichiers tests/application_test_empty_rows.csv"
    
    # Ouvrir le fichier et l'envoyer à l'API et récupérer la réponse
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/csv')}
        response = requests.post(url, files=files)
    
    # Enregistrer les détails du test
    logger.info(test_name)
    logger.info(f"Status code: {response.status_code}")
    logger.info(f"Status détail: {response.json()['detail']}")

    # Assert pour validation pytest
    assert response.status_code == 400, f"Attendu 400, reçu {response.status_code}"

##################################### 2. Test fichier contenant une ligne totalement vide #####################################

def test_fichier_une_ligne_vide():
    """Test fichier une ligne totalement vide"""
    # Définir l'url, le nom du test et le lien du fichier à utiliser
    url = "http://localhost:8000/predict"
    test_name = "Test fichier une ligne totalement vide"
    file_path = "fichiers tests/application_test_one_empty_row.csv"
    
    # Ouvrir le fichier et l'envoyer à l'API et récupérer la réponse
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/csv')}
        response = requests.post(url, files=files)
    
    # Enregistrer les détails du test
    logger.info(test_name)
    logger.info(f"Status code: {response.status_code}")
    logger.info(f"Status détail: {response.json()['detail']}")

    # Assert pour validation pytest
    assert response.status_code == 400, f"Attendu 400, reçu {response.status_code}"

######################################## 3. Test fichier avec des colonnes manquantes #########################################

def test_fichier_colonnes_manquantes():
    """Test fichier avec des colonnes manquantes"""
    # Définir l'url, le nom du test et le lien du fichier à utiliser
    url = "http://localhost:8000/predict"
    test_name = "Test fichier avec des colonnes manquantes"
    file_path = "fichiers tests/application_test_missing_columns.csv"
    
    # Ouvrir le fichier et l'envoyer à l'API et récupérer la réponse
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/csv')}
        response = requests.post(url, files=files)
    
    # Enregistrer les détails du test
    logger.info(test_name)
    logger.info(f"Status code: {response.status_code}")
    logger.info(f"Status détail: {response.json()['detail']}")

    # Assert pour validation pytest
    assert response.status_code == 400, f"Attendu 400, reçu {response.status_code}"

#################################################### 4. Test fichier xlsx #####################################################

def test_fichier_xlsx():
    """Test fichier xlsx"""
    # Définir l'url, le nom du test et le lien du fichier à utiliser
    url = "http://localhost:8000/predict"
    test_name = "Test fichier xlsx"
    file_path = "fichiers tests/application_test_2_clients.xlsx"
    
    # Ouvrir le fichier et l'envoyer à l'API et récupérer la réponse
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/csv')}
        response = requests.post(url, files=files)
    
    # Enregistrer les détails du test
    logger.info(test_name)
    logger.info(f"Status code: {response.status_code}")
    logger.info(f"Status détail: {response.json()['detail']}")

    # Assert pour validation pytest
    assert response.status_code == 422, f"Attendu 422, reçu {response.status_code}"

############################################ 5. Test fichier beaucoup trop lourd ##############################################

def test_fichier_trop_lourd():
    """Test fichier beaucoup trop lourd"""
    # Définir l'url, le nom du test et le lien du fichier à utiliser
    url = "http://localhost:8000/predict"
    test_name = "Test fichier beaucoup trop lourd"
    file_path = "fichiers tests/application_test_heavy_file.csv"
    
    # Ouvrir le fichier et l'envoyer à l'API et récupérer la réponse
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/csv')}
        response = requests.post(url, files=files)
    
    # Enregistrer les détails du test
    logger.info(test_name)
    logger.info(f"Status code: {response.status_code}")
    logger.info(f"Status détail: {response.json()['detail']}")

    # Assert pour validation pytest
    assert response.status_code == 413, f"Attendu 413, reçu {response.status_code}"

#################################################### 6. Test fichier vide #####################################################

def test_fichier_vide():
    """Test fichier vide"""
    # Définir l'url, le nom du test et le lien du fichier à utiliser
    url = "http://localhost:8000/predict"
    test_name = "Test fichier vide"
    file_path = "fichiers tests/application_test_empty_file.csv"
    
    # Ouvrir le fichier et l'envoyer à l'API et récupérer la réponse
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/csv')}
        response = requests.post(url, files=files)
    
    # Enregistrer les détails du test
    logger.info(test_name)
    logger.info(f"Status code: {response.status_code}")
    logger.info(f"Status détail: {response.json()['detail']}")

    # Assert pour validation pytest
    assert response.status_code == 400, f"Attendu 400, reçu {response.status_code}"

######################################## 7. Test fichier avec nombre au format texte ##########################################

def test_fichier_format_erroné():
    """Test fichier avec nombre au format texte"""
    # Définir l'url, le nom du test et le lien du fichier à utiliser
    url = "http://localhost:8000/predict"
    test_name = "Test fichier avec nombre au format texte"
    file_path = "fichiers tests/application_test_wrong_dtype.csv"
    
    # Ouvrir le fichier et l'envoyer à l'API et récupérer la réponse
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/csv')}
        response = requests.post(url, files=files)
    
    # Enregistrer les détails du test
    logger.info(test_name)
    logger.info(f"Status code: {response.status_code}")

    # Assert pour validation pytest
    assert response.status_code == 200, f"Attendu 200, reçu {response.status_code}"

#################### 8. Test fichier avec du texte dans une colonne qui devrait ne contenir que des nombres ###################

def test_fichier_valeur_incohérente():
    """Test fichier avec du texte dans une colonne qui devrait ne contenir que des nombres"""
    # Définir l'url, le nom du test et le lien du fichier à utiliser
    url = "http://localhost:8000/predict"
    test_name = "Test fichier avec du texte dans une colonne qui devrait ne contenir que des nombres"
    file_path = "fichiers tests/application_test_wrong_values.csv"
    
    # Ouvrir le fichier et l'envoyer à l'API et récupérer la réponse
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/csv')}
        response = requests.post(url, files=files)
    
    # Enregistrer les détails du test
    logger.info(test_name)
    logger.info(f"Status code: {response.status_code}")
    logger.info(f"Status détail: {response.json()['detail']}")

    # Assert pour validation pytest
    assert response.status_code == 400, f"Attendu 400, reçu {response.status_code}"

##################################################### 9. Test sans fichier ####################################################

def test_sans_fichier():
    """Test sans fichier"""
    # Définir l'url
    url = "http://localhost:8000/predict"
    test_name = "Test sans fichier"
    
    # Envoyer une requête à l'API sans fichier
    response = requests.post(url)
    
    # Enregistrer les détails du test
    logger.info(test_name)
    logger.info(f"Status code: {response.status_code}")
    logger.info(f"Status détail: {response.json()['detail']}")

    # Assert pour validation pytest
    assert response.status_code == 422, f"Attendu 422, reçu {response.status_code}"