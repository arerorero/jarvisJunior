import numpy as np
import cv2
import mysql.connector
import pyautogui
import subprocess


def installAll():
    installs = [
    'pip install numpy',
    'pip install opencv-python',
    'pip install mysql-connector-python',
    'pip install pyautogui'
]
    for key in installs:
        subprocess.check_call(key)

def linhas():
    mario = cv2.imread('assets/mario.jpg',0)
    gray = cv2.cvtColor(mario,cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 100, 0.0001, 30)
    corners = np.int0(corners)

    for corner in corners:
        x, y = corner.ravel() 
        cv2.circle(mario, (x,y), 3, (255,0,0), -1)

    for i in range(len(corners)):
        for j in range(i+1,len(corners)):
            corner1 = tuple(corner[i][0])
            corner2 = tuple(corner[j][0])
            color = tuple(map(lambda x: int(x), np.random.randint(0,255,size=3)))
            cv2.line(mario,corner1,corner2,color,1)

def moveMouse(loc):
    x,y = loc

    pyautogui.moveTo(x, y)

def find_image_position(gray_b):
    # TESTAR MAIS
    if gray_b.dtype != np.uint8:
        gray_b = gray_b.astype(np.uint8)
    if gray_b.ndim >= 2:
        gray_b = cv2.cvtColor(gray_b, cv2.COLOR_BGR2GRAY)
    # Convert as imagens para escala de cinza
    gray_a = cv2.imread('assets/mario.jpg',0)

    # Aplica o método de correspondência
    result = cv2.matchTemplate(gray_b, gray_a, cv2.TM_CCOEFF_NORMED)

    # Define um limite para considerar uma correspondência
    threshold = 0.01

    # Encontra as correspondências acima do limite
    locations = np.where(result >= threshold)

    if locations[0].size > 0 and locations[1].size > 0:
        # Encontrou uma correspondência
        top_left = (locations[1][0], locations[0][0])
        bottom_right = (top_left[0] + gray_a.shape[1], top_left[1] + gray_a.shape[0])
        return top_left, bottom_right
    else:
        # Não encontrou correspondência
        return None
    
def detecImg():
        
    methods = [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED,cv2.TM_CCORR,
                cv2.TM_CCORR_NORMED,cv2.TM_CCOEFF,cv2.TM_CCOEFF_NORMED]

    mario64 =  cv2.imread('assets/mario.jpg')
    mario =  cv2.imread('assets/mario.jpg',0)
    marioName = cv2.imread('assets/marioName.jpg',0)
    h, w = marioName.shape

    for method in methods:
        mario2 = mario.copy()
        result = cv2.matchTemplate(mario2,marioName,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
            local = min_loc
        else:
            local = max_loc
        bottom_right = (local[0]+w, local[1]+h)
        cv2.rectangle(mario64,local,bottom_right,255,5)
        cv2.waitKey(0) == ord('q')
        cv2.destroyAllWindows
        
def camera():
    cam = cv2.VideoCapture(0)

    while cam.isOpened():
        ret, frame = cam.read()
        if ret:
            cv2.imshow('cam', frame)

            #69
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break
        
    cam.release()
    cv2.destroyAllWindows()

def connector(query,values):
    # "INSERT INTO imagens (nome_arquivo, conteudo) VALUES (%s, %s)" exemplo de insert
    dataBase = {
        # 69
    'host': 'mysql.ciarama.com.br',
    'user': 'ciarama05',
    'password': 'Cavalo200posto',
    'database': 'ciarama05'
    }
    try:
        conn = mysql.connector.connect(
        host=dataBase['host'],
        user=dataBase['user'],
        password=dataBase['password'],
        database=dataBase['database']
    ) 
        print("Conexão bem-sucedida!")
        cursor = conn.cursor()
        
        cursor.execute(query,values)
        
        result = cursor.fetchone()
        
        cursor.close()
        conn.commit()
        conn.close()
        print("Conexão encerrada.")
        if result:
            return result[0]
    
    except mysql.connector.Error as error:
        print("Erro ao conectar ao banco de dados: {}".format(error))

def tela():
    while True:
        
        screen = pyautogui.screenshot(region=(0, 0, 1920, 1280))
        image = cv2.cvtColor(np.array(screen),0)
        
        print(find_image_position(image))
        
        if cv2.waitKey(1) == ord('q'):
            break


def verificar_imagem_contida(imagem_principal, imagem_procurada):
    # Carrega as imagens
    img_principal = cv2.imread(imagem_principal, cv2.IMREAD_GRAYSCALE)
    print(img_principal.shape)
    img_procurada = cv2.imread(imagem_procurada, cv2.IMREAD_GRAYSCALE)
    print(img_procurada.shape)
    return
    # Inicializa o detector ORB
    orb = cv2.ORB_create()

    # Detecta e computa as características das duas imagens
    kp1, des1 = orb.detectAndCompute(img_principal, None)
    kp2, des2 = orb.detectAndCompute(img_procurada, None)

    # Inicializa o matcher de correspondência de características
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Encontra as correspondências das características
    matches = bf.match(des1, des2)

    # Ordena as correspondências por distância
    matches = sorted(matches, key=lambda x: x.distance)

    # Verifica se há correspondências suficientes para considerar a imagem como contida
    if len(matches) > 10:
        return True
    else:
        return False

# Exemplo de uso
imagem_principal = 'assets/mario.jpg'
imagem_procurada = 'assets/marioName.jpg'

if verificar_imagem_contida(imagem_principal, imagem_procurada):
    print("A imagem procurada está contida na imagem principal.")
else:
    print("A imagem procurada não está contida na imagem principal.")
    