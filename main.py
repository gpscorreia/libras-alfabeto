import cv2
import mediapipe as mp
import urllib.request
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def download_model():
    model_path = "hand_landmarker.task"

    if not os.path.exists(model_path):
        print("Baixando modelo Hand Landmarker...")

        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/"
            "hand_landmarker.task"
        )

        urllib.request.urlretrieve(url, model_path)

        print("Download concluído!")

    return model_path


def finger_is_extended(landmarks, tip, pip):
    return landmarks[tip].y < landmarks[pip].y


def thumb_is_extended(landmarks):
    return abs(landmarks[4].x - landmarks[5].x) > abs(landmarks[3].x - landmarks[5].x)


def main():
    model_path = download_model()

    # Configuração do MediaPipe Hand Landmarker
    base_options = python.BaseOptions(model_asset_path=model_path)

    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)

    detector = vision.HandLandmarker.create_from_options(options)

    # Abrir webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao acessar a webcam.")
        detector.close()
        return

    print("Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Não foi possível capturar o frame.")
            break

        # Espelhar a imagem
        frame = cv2.flip(frame, 1)

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Criar imagem para o MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detectar mãos
        detection_result = detector.detect(mp_image)

        # Verificar se encontrou uma mão
        if detection_result.hand_landmarks:

            cv2.putText(
                frame,
                "Mao detectada",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Percorrer as mãos detectadas
            for hand_landmarks in detection_result.hand_landmarks:

                thumb_extended = thumb_is_extended(hand_landmarks)

                index_extended = finger_is_extended(hand_landmarks, 8, 6)

                middle_extended = finger_is_extended(hand_landmarks, 12, 10)

                ring_extended = finger_is_extended(hand_landmarks, 16, 14)

                pinky_extended = finger_is_extended(hand_landmarks, 20, 18)

                print(
                    "Polegar:",
                    thumb_extended,
                    " Indicador:",
                    index_extended,
                    " Médio:",
                    middle_extended,
                    " Anelar:",
                    ring_extended,
                    " Mindinho:",
                    pinky_extended,
                )

                # Desenhar os 21 landmarks
                for landmark in hand_landmarks:

                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Mostrar o número de landmarks
                cv2.putText(
                    frame,
                    f"Landmarks: {len(hand_landmarks)}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

        # Mostrar imagem
        cv2.imshow("Alfabeto em Libras", frame)

        # Q para sair
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
