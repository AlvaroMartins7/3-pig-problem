from ultralytics import YOLO
import cv2

# Carregar o modelo treinado
#model = YOLO("experimentos/no_noise/detect/train/weights/best.pt")
model = YOLO("experimentos/noise/detect/train/weights/best.pt")


# Processar o vídeo em modo streaming
results = model("12-Ceiling_Cam.mp4", stream=True)

# Definir a escala para redimensionar o vídeo (ex.: 50% do tamanho original)
resize_scale = 0.5  # Altere este valor conforme necessário

# Processar cada quadro
for result in results:
    frame = result.plot()  # Adiciona as detecções no quadro

    # Redimensionar o quadro
    frame_resized = cv2.resize(frame, (1280, 720))

    # Exibir o quadro redimensionado
    cv2.imshow("Detections", frame_resized)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
