#app.py
import os
import cv2
import uuid
import json
import time
import requests
from collections import Counter
import logging
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = '/Users/gimdonghun/Documents/chordswap/webChordS/uploadsReal'
PROCESSED_FOLDER = '/Users/gimdonghun/Documents/chordswap/webChordS/processedReal'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# 로깅 설정 (콘솔에 출력하기 위해)
logging.basicConfig(level=logging.INFO, format='%(message)s')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

def call_ocr_api(image_path, api_url, secret_key):
    try:
        with open(image_path, 'rb') as image_file:
            files = [('file', image_file)]
            request_json = {
                'images': [{'format': 'jpg', 'name': 'demo'}],
                'requestId': str(uuid.uuid4()),
                'version': 'V2',
                'timestamp': int(round(time.time() * 1000))
            }
            payload = {'message': json.dumps(request_json).encode('UTF-8')}
            headers = {'X-OCR-SECRET': secret_key}
            
            response = requests.post(api_url, headers=headers, data=payload, files=files)
            response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
            return response.json()
    except Exception as e:
        logging.error(f"OCR API 호출 중 오류 발생: {e}")
        return None

def process_image(image_path, ocr_result, half_steps, save_directory=PROCESSED_FOLDER):
    try:
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"이미지를 읽을 수 없습니다: {image_path}")
            return None

        roi_img = img.copy()
        font_italic = cv2.FONT_ITALIC
        
        chords = []  # 코드를 저장할 리스트

        for field in ocr_result['images'][0]['fields']:
            text = field['inferText']
            first_char = text[0] if len(text) > 0 else ''

            if first_char.isupper() and first_char in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                vertices_list = field['boundingPoly']['vertices']
                pts = [tuple(vertice.values()) for vertice in vertices_list]
                topLeft = [int(_) for _ in pts[0]]
                bottomRight = [int(_) for _ in pts[2]]

                fill_img = cv2.rectangle(roi_img, tuple(topLeft), tuple(bottomRight), (255, 255, 255), thickness=-1)
                modified_text = modify_text(text)
                
                transposed_chord = transpose_chord(modified_text, half_steps)
                
                # 변환 과정 로그 출력 (간단하게)
                logging.info(f"{modified_text} -> {transposed_chord}")
                
                new_code = cv2.putText(fill_img, transposed_chord, (topLeft[0], topLeft[1] + 10), font_italic, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                
                chords.append(transposed_chord)  # 변환된 코드를 리스트에 추가

        # 키 추정
        estimated_key = estimate_key_by_frequency(chords)

        # 파일 확장자 확인 및 설정
        _, file_extension = os.path.splitext(image_path)
        if file_extension == '':
            file_extension = '.jpg'  # 기본 확장자 설정

        # 저장될 파일 경로 설정
        processed_image_filename = f'processed_{os.path.splitext(os.path.basename(image_path))[0]}{file_extension}'
        processed_image_path = os.path.join(save_directory, processed_image_filename)
        
        cv2.imwrite(processed_image_path, roi_img)
        logging.info(f"변환된 이미지가 저장되었습니다: {processed_image_path}")
        
        # 변환된 이미지 파일명과 추정된 키를 반환
        return processed_image_filename, estimated_key
    except Exception as e:
        logging.error(f"이미지 처리 중 오류 발생: {e}")
        return None, None

def modify_text(text):
    # '#'를 인식하지 못하는 문제 해결
    text = text.replace("t", "#").replace("&", "#")
    # 'm'과 'sus' 사이에 공백이 있을 경우 제거
    text = text.replace("m 7", "m7").replace("sus 4", "sus4")
    return text

def use_flat_notation(note):
    flat_notes = ['Bb', 'Db', 'Eb', 'Gb', 'Ab']
    return note in flat_notes

def transpose_chord(chord, half_steps):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    flat_notes = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

    def transpose_note(note, steps):
        if note in notes:
            index = notes.index(note)
        elif note in flat_notes:
            index = flat_notes.index(note)
        else:
            return note  # 변경할 수 없는 경우 원래 노트 반환

        new_index = (index + steps) % 12
        new_note = flat_notes[new_index] if use_flat_notation(flat_notes[new_index]) else notes[new_index]
        return new_note

    # 코드 파싱
    parts = chord.split('/')
    root = parts[0]
    bass = parts[1] if len(parts) > 1 else None

    # 루트 노트 변환
    new_root = root[:1]  # 첫 글자(노트)
    if len(root) > 1 and root[1] in ['#', 'b']:
        new_root += root[1]  # 샵이나 플랫 포함
    new_root = transpose_note(new_root, half_steps)
    
    # 나머지 부분 (7, m, sus4 등) 유지
    remainder = root[len(new_root):]
    new_chord = new_root + remainder

    # 베이스 노트가 있으면 변환
    if bass:
        new_bass = transpose_note(bass, half_steps)
        new_chord += '/' + new_bass

    return new_chord

def estimate_key_by_frequency(chords):
    chord_counts = Counter(chords)
    most_common_chord, _ = chord_counts.most_common(1)[0]
    root_note = most_common_chord.split('/')[0]  # 베이스 노트 제거
    root_note is root_note.rstrip('m7sus4')  # 코드 타입 제거
    
    major_keys = {
        'C': 'C', 'C#': 'C#', 'D': 'D', 'D#': 'D#', 'E': 'E', 'F': 'F', 'F#': 'F#', 
        'G': 'G', 'G#': 'G#', 'A': 'A', 'A#': 'A#', 'B': 'B'
    }
    
    for key, root in major_keys.items():
        if root_note.startswith(root):
            return key
    return None

@app.route('/')
def index():
    original_image = request.args.get('original_image')
    processed_image = request.args.get('processed_image')
    estimated_key = request.args.get('estimated_key')
    return render_template('index.html', original_image=original_image, processed_image=processed_image, estimated_key=estimated_key)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    half_steps = request.form.get('half_steps', type=int, default=0)
    if file:
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            logging.info(f"원본 이미지가 저장되었습니다: {file_path}")
            
            api_url = 'API'
            secret_key = 'secretKey'

            ocr_result = call_ocr_api(file_path, api_url, secret_key)
            if ocr_result is None:
                return 'OCR 처리 중 오류가 발생했습니다.'

            processed_image_path, estimated_key = process_image(file_path, ocr_result, half_steps)

            if processed_image_path:
                return redirect(url_for('index', original_image=filename, processed_image=os.path.basename(processed_image_path), estimated_key=estimated_key))
            else:
                return '이미지 처리 중 오류가 발생했습니다.'
        except Exception as e:
            logging.error(f"파일 업로드 중 오류 발생: {e}")
            return '파일 업로드 중 오류가 발생했습니다.'

@app.route('/uploadsReal/<filename>')
def uploaded_file(filename):
    print(f"Requesting uploaded file: {filename}")
    print(f"Full path: {os.path.join(app.config['UPLOAD_FOLDER'], filename)}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processedReal/<filename>')
def processed_file(filename):
    print(f"Requesting processed file: {filename}")
    print(f"Full path: {os.path.join(app.config['PROCESSED_FOLDER'], filename)}")
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


print(f"UPLOAD_FOLDER: {os.path.abspath(UPLOAD_FOLDER)}")
print(f"PROCESSED_FOLDER: {os.path.abspath(PROCESSED_FOLDER)}")

if __name__ == '__main__':
    app.run(debug=True, port=8080)
