
import cv2
import numpy as np
import os
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="TnW57F8wwErlqd3pWkXY"
)


def find_board_squares_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image from {image_path}")
        return None, None
    
    top_left = (560, 140)
    top_right = (1360, 140)
    bottom_left = (560, 940)
    bottom_right = (1360, 940)

    input_points = np.float32([top_left, top_right, bottom_right, bottom_left])
    output_points = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
    M = cv2.getPerspectiveTransform(input_points, output_points)
    
    squares = {}
    square_size = 400 / 8
    
    file_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    rank_names = ['8', '7', '6', '5', '4', '3', '2', '1']
    
    for rank_idx, rank in enumerate(rank_names):
        for file_idx, file in enumerate(file_names):
            square_name = file + rank
            x1 = int(file_idx * square_size)
            y1 = int(rank_idx * square_size)
            x2 = int(x1 + square_size)
            y2 = int(y1 + square_size)
            squares[square_name] = (x1, y1, x2, y2)
    
    return squares, M


def find_pieces_on_board_from_robowflow(image_path, board_squares, perspective_matrix):
    try:
        result = CLIENT.infer(image_path, model_id="chess-piece-detection-lnpzs/1")
    except Exception as e:
        print(f"Error during Roboflow inference for {image_path}: {e}")
        return None, None

    current_board_state = {}
    detections_list = []
    
    for detection in result.get('predictions', []):
        piece_class = detection['class']
        center_x, center_y = detection['x'], detection['y']

        points_original = np.float32([[[center_x, center_y]]])
        points_warped = cv2.perspectiveTransform(points_original, perspective_matrix)
        warped_x, warped_y = points_warped[0][0]
        
        for square_name, (sq_x1, sq_y1, sq_x2, sq_y2) in board_squares.items():
            if sq_x1 <= warped_x < sq_x2 and sq_y1 <= warped_y < sq_y2:
                current_board_state[square_name] = piece_class
                detections_list.append((piece_class, square_name, detection['x'], detection['y'], detection['width'], detection['height']))
                break
                
    return current_board_state, detections_list


def create_minimap(current_state):
    minimap_size = 400
    square_size = minimap_size // 8
    minimap = np.zeros((minimap_size, minimap_size, 3), dtype=np.uint8)
    
    
    white_square_color = (200, 200, 200)
    black_square_color = (100, 100, 100)
    
 
    for row in range(8):
        for col in range(8):
            color = white_square_color if (row + col) % 2 == 0 else black_square_color
            cv2.rectangle(minimap, (col * square_size, row * square_size), 
                          ((col + 1) * square_size, (row + 1) * square_size), color, -1)
    
   
    piece_colors = {
        'white-pawn': (255, 255, 255), 'white-knight': (255, 255, 255),
        'white-bishop': (255, 255, 255), 'white-rook': (255, 255, 255),
        'white-queen': (255, 255, 255), 'white-king': (255, 255, 255),
        'black-pawn': (0, 0, 0), 'black-knight': (0, 0, 0),
        'black-bishop': (0, 0, 0), 'black-rook': (0, 0, 0),
        'black-queen': (0, 0, 0), 'black-king': (0, 0, 0)
    }
    
    file_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    rank_names = ['8', '7', '6', '5', '4', '3', '2', '1']
    
    for square_name, piece_class in current_state.items():
        file = file_names.index(square_name[0])
        rank = rank_names.index(square_name[1])
        
        center_x = int((file + 0.5) * square_size)
        center_y = int((rank + 0.5) * square_size)
        print()
        cv2.circle(minimap, (center_x, center_y), square_size // 3, piece_colors.get(piece_class, (255, 0, 0)), -1)
        
    return minimap


def main():
    image_folder = '/yolo/raw_video_frames2'
    
   
    FRAME_STEP = 20
    OUTPUT_VIDEO_PATH = '/yolo/video.mp4'

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
    if not image_files:
        print("No image files found in the specified folder.")
        return

     
    first_image_path = os.path.join(image_folder, image_files[0])
    board_squares, perspective_matrix = find_board_squares_from_image(first_image_path)
    if board_squares is None:
        return

     
    sample_image = cv2.imread(first_image_path)
    height, width, _ = sample_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 20.0, (width + 200, height))

    print("Starting processing of raw images and video creation...")
    
    previous_state, _ = find_pieces_on_board_from_robowflow(first_image_path, board_squares, perspective_matrix)
    
    last_move_text = ""

    for i in range(len(image_files)):
        
        if i % FRAME_STEP != 0 and i != len(image_files) - 1:
            continue
        
        current_image_path = os.path.join(image_folder, image_files[i])
        current_state, detections_list = find_pieces_on_board_from_robowflow(current_image_path, board_squares, perspective_matrix)
        
        if current_state is None:
            continue

        moved_from, moved_to = None, None
        
        
        if previous_state is not None:
            for square, piece in previous_state.items():
                if square not in current_state or current_state.get(square) != piece:
                    moved_from = square
            
            for square, piece in current_state.items():
                if square not in previous_state or previous_state.get(square) != piece:
                    moved_to = square

        current_frame = cv2.imread(current_image_path)
        if current_frame is None:
            continue

        
        if moved_from and moved_to:
            piece_moved = previous_state.get(moved_from, "unknown piece")
            last_move_text = f"{piece_moved} from {moved_from} to {moved_to}"
            print(f"Detected move: {last_move_text} (Frame {i-FRAME_STEP} -> {i})")

            # پیدا کردن مختصات مهره جابه‌جا شده برای رسم مربع
            for detection in detections_list:
                piece_class, square_name, x, y, w, h = detection
                if square_name == moved_from:
                    color = (0, 0, 255) if 'white' in piece_moved else (255, 0, 0)
                    cv2.rectangle(current_frame, (int(x - w/2), int(y - h/2)), 
                                  (int(x + w/2), int(y + h/2)), color, 5)
                elif square_name == moved_to:
                    color = (0, 255, 0) if 'white' in piece_moved else (0, 0, 0)
                    cv2.rectangle(current_frame, (int(x - w/2), int(y - h/2)), 
                                  (int(x + w/2), int(y + h/2)), color, 5)
        
        
        if last_move_text:
            cv2.putText(current_frame, last_move_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        
        minimap = create_minimap(current_state)
        combined_frame = np.zeros((height, width + 200, 3), dtype=np.uint8)
        combined_frame[:, :width] = current_frame
        combined_frame[100:100+200, width:width+200] = minimap
        
        out.write(combined_frame)
        previous_state = current_state

    out.release()
    print("Video creation finished.")

if __name__ == '__main__':
    main()
