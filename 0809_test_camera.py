import cv2 as cv
import os
from datetime import datetime
from tkinter import Tk, Button, Label, filedialog, StringVar, Frame, Toplevel, Canvas, Scrollbar
from PIL import Image, ImageTk

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 카메라 찾기 함수
def find_working_camera():
    index = 1
    while True:
        cap = cv.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"카메라 연결 성공: 인덱스 {index}")
                return cap
            else:
                cap.release()
        index += 1
        if index > 10:
            print("사용 가능한 카메라를 찾을 수 없습니다.")
            return None

# 이미지 저장 함수
def save_image(cam, folder_path):
    ret, frame = cam.read()
    if ret:
        now = datetime.now()
        timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
        filename = os.path.join(folder_path, f"image_{timestamp}.jpg")
        cv.imwrite(filename, frame)
        print(f"이미지 저장 완료: {filename}")
        return filename
    else:
        print("이미지 저장 실패")
        return None

# 실시간 카메라 프레임 업데이트 함수
def update_frame(cam, camera_label, callback):
    ret, frame = cam.read()
    if ret:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
    camera_label.after(10, update_frame, cam, camera_label, callback)

# 카메라 촬영 윈도우 생성 함수
def open_camera_window(image_label, folder_path):
    camera_window = Toplevel()
    camera_window.title("카메라 촬영")
    camera_label = Label(camera_window)
    camera_label.pack()
    
    def capture_and_close():
        filename = save_image(cap, folder_path)
        if filename:
            img = Image.open(filename)
            img = img.resize((500, 500), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            image_label.imgtk = imgtk
            image_label.configure(image=imgtk)
            display_analysis_results(image_label, filename)  # 분석 결과 출력
        camera_window.destroy()
    
    capture_button = Button(camera_window, text="촬영", command=capture_and_close)
    capture_button.pack()
    
    update_frame(cap, camera_label, capture_and_close)
    camera_window.mainloop()

# 이미지 선택 함수
def select_image(image_label):
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((500, 500), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        image_label.imgtk = imgtk
        image_label.configure(image=imgtk)
        display_analysis_results(image_label, file_path)  # 분석 결과 출력

# 빈 칸 클릭 이벤트 핸들러
def on_image_slot_click(image_label):
    option_window = Toplevel()
    option_window.title("옵션 선택")
    option_window.geometry("300x200")  # 창 크기 고정

    def capture_image_and_close():
        option_window.destroy()
        open_camera_window(image_label, folder_path)
        #option_window.destroy()

    def select_image_and_close():
        select_image(image_label)
        option_window.destroy()

    capture_button = Button(option_window, text="촬영", command=capture_image_and_close)
    capture_button.pack(pady=10)

    select_button = Button(option_window, text="갤러리에서 선택", command=select_image_and_close)
    select_button.pack(pady=10)

    option_window.mainloop()

# 분석 윈도우 생성 함수
# 분석 윈도우 생성 함수
def open_analysis_window():
    analysis_window = Toplevel()
    analysis_window.title("분석")
    analysis_window.geometry("1200x800")

    # 캔버스와 스크롤바 생성
    canvas = Canvas(analysis_window)
    scrollbar = Scrollbar(analysis_window, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollable_frame = Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # 이미지 프레임을 중앙에 배치
    for i in range(5):
        frame = Frame(scrollable_frame)
        frame.pack(pady=10, padx=10, expand=True, fill="both")
        
        image_label = Label(frame, text="이미지 없음", width=500, height=500, relief="solid")
        image_label.pack()
        image_label.bind("<Button-1>", lambda e, lbl=image_label: on_image_slot_click(lbl))
    
    analysis_window.mainloop()

#이미지 전처리
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(500, 500))  # 모델에 맞는 크기로 조정
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    img_array /= 255.0  # 정규화
    return img_array

def analyze_image(img_path):
    processed_image = preprocess_image(img_path)
    predictions = model.predict(processed_image)
    return predictions  # 예측 결과 반환

def interpret_predictions(predictions):
    class_labels = ["양호", "경증", "중증"]  # 클래스 이름 예시
    predicted_class = np.argmax(predictions)
    return f"Predicted: {class_labels[predicted_class]} ({predictions[0][predicted_class]:.2f})"

# 기존 이미지 선택 또는 촬영 후
def display_analysis_results(image_label, img_path):
    predictions = analyze_image(img_path)
    result_text = interpret_predictions(predictions)  # 예측 결과 해석
    result_label = Label(image_label.master, text=result_text)
    result_label.pack()

# 이미지 선택 또는 촬영 후 호출







#모델 로드
model = load_model('C:\\Users\\main\\Desktop\\saveimg\\scalp_classification_model11.h5')

# GUI 초기화
root = Tk()
root.title("메인 메뉴")
root.geometry("400x300")

# 분석 버튼
analysis_button = Button(root, text="분석", command=open_analysis_window)
analysis_button.pack(pady=10)

# 기록 버튼
record_button = Button(root, text="기록")
record_button.pack(pady=10)

# 카메라 연결
cap = find_working_camera()
if not cap:
    print("카메라를 찾을 수 없습니다.")
else:
    print("카메라 연결됨")

# 폴더 경로 초기화
folder_path = "C:\\Users\\main\\Desktop\\saveimg"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# GUI 실행
root.mainloop()

# 카메라 해제 및 모든 창 닫기
if cap:
    cap.release()
cv.destroyAllWindows()
