import cv2
import numpy as np
import easyocr
import pyvista as pv
import re
import math
import os
import json
from PIL import Image
from PIL.Image import Resampling # For Image.Resampling.LANCZOS
from sklearn.cluster import DBSCAN


class FloorPlanConverter:
    def __init__(self):
        
        self.trocr_processor = None
        self.trocr_model = None

        import os

        USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"

        try:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=USE_GPU)
            print(f"EasyOCR loaded with GPU={USE_GPU}")

        except Exception as e:
            print(f"EasyOCR GPU failed: {e}")

            try:
               self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
               print("Fallback to CPU successful.")
            except Exception as e_cpu:
               print(f"CPU fallback failed: {e_cpu}")

        self.image_path = None
        self.original_image_pil = None 
        self.displayed_image_pil = None 

        self.room_dimensions = {}
        self.walls = [] 
        self.curved_walls = [] 
        self.scale_factor = 1.0 
        self.display_scale_factor = 1.0 
        self.default_height = 9.0 
        self.wall_thickness = 0.5 
        self.room_positions = {}
        
        self.door_width_default = 2.8
        self.door_height_default = 6.8
        self.window_width_default = 3.0
        self.window_height_default = 4.0
        self.window_sill_default = 3.0

        self.materials = {
            "floor": "#FFFFFF",
            "wall": "#0041C2",
            "ceiling": "#F3D2F1", 
            "door_frame": "#A0522D", "door_panel": "#8B4513",
            "window_frame": "#A0522D", "window_glass": "#ADD8E6",
            "furniture": {
                "bed": "#4682B4", "nightstand": "#8B4513", "counter": "#D3D3D3", "island": "#A9A9A9",
                "sofa": "#6B8E23", "table": "#8B4513", "chair": "#CD853F", "bathtub": "#B0E0E6",
                "toilet": "#F0F8FF", "sink": "#F5F5F5", "wardrobe": "#8B4513", "tv_stand": "#A9A9A9",
                "bookshelf": "#8B4513", "desk": "#D3D3D3", "oven": "#696969", "refrigerator": "#FFFFFF",
            }
        }
        
        self.label_font_size = 14
        self.show_labels_in_3d = False 
        self.selection_rect = None
        self.start_x_canvas = None
        self.start_y_canvas = None
        
    def determine_room_type(self, room_name):
        room_name_lower = room_name.lower()
        if "bed" in room_name_lower: return "Bedroom"
        if "kitch" in room_name_lower: return "Kitchen"
        if "living" in room_name_lower: return "Living Room"
        if "bath" in room_name_lower: return "Bathroom"
        if "dining" in room_name_lower: return "Dining Room"
        if "office" in room_name_lower: return "Office"
        if "hall" in room_name_lower: return "Other" 
        return "Other"

    def process_current_image(self):
        
        try:
            cv_original_image = np.array(self.original_image_pil.convert('RGB'))
            cv_original_image = cv2.cvtColor(cv_original_image, cv2.COLOR_RGB2BGR)
            
            image_for_text_masking = cv_original_image.copy()
            image_for_wall_detection = cv_original_image.copy()

            if self.scale_factor == 1.0: 
                 try:
                    scale_input = 10.0
                    if scale_input:
                        new_scale = float(scale_input)
                        if new_scale <= 0: raise ValueError("Scale must be positive")
                        self.scale_factor = new_scale
                 except ValueError:
                    print("Scale Warning", f"Invalid scale input. Using existing scale: {self.scale_factor:.2f} px/ft.")
            
            if self.easyocr_reader:
               
                try:
                    gray_for_mask_ocr = cv2.cvtColor(image_for_text_masking, cv2.COLOR_BGR2GRAY)
                    ocr_results_for_masking = self.easyocr_reader.readtext(gray_for_mask_ocr, detail=1, paragraph=False)
                    for (bbox, text, prob) in ocr_results_for_masking:
                        if prob < 0.3: continue 
                        points = np.array(bbox, dtype=np.int32)
                        rect_x_min = np.min(points[:, 0]) - 2 
                        rect_y_min = np.min(points[:, 1]) - 2
                        rect_x_max = np.max(points[:, 0]) + 2
                        rect_y_max = np.max(points[:, 1]) + 2
                        cv2.rectangle(image_for_wall_detection, (rect_x_min, rect_y_min), (rect_x_max, rect_y_max), 
                                      (255, 255, 255), -1) 
                except Exception as e:
                    print(f"Error during OCR for text masking: {e}. Wall detection might be affected.")
                   
            self.walls = self.detect_walls(image_for_wall_detection) 
            self.curved_walls = self.detect_curved_walls(image_for_wall_detection) 

            # --- Manual Injection of Openings for Demonstration ---
            if len(self.walls) > 0 and self.scale_factor > 0:
                walls_with_openings_added = 0
                for i, wall_data in enumerate(self.walls):
                    if walls_with_openings_added >= 3 : break # Limit openings for demo
                    
                    wall_length_ft = wall_data["length"] / self.scale_factor
                    if wall_length_ft > 5: 
                        if "openings" not in wall_data: wall_data["openings"] = []
                        
                        wall_data["openings"].append({
                            "position_on_wall": wall_data["length"] * 0.5, 
                            "width_px": self.door_width_default * self.scale_factor,
                            "height_px": self.door_height_default * self.scale_factor,
                            "sill_px": 0, 
                            "type": "door"
                        })
                        
                        if wall_length_ft > self.door_width_default + self.window_width_default + 3: 
                            wall_data["openings"].append({
                                "position_on_wall": wall_data["length"] * 0.20, 
                                "width_px": self.window_width_default * self.scale_factor,
                                "height_px": self.window_height_default * self.scale_factor,
                                "sill_px": self.window_sill_default * self.scale_factor,
                                "type": "window"
                            })
                        print(f"Manually added sample openings to wall index {i} (length: {wall_length_ft:.1f} ft)")
                        walls_with_openings_added +=1
            # --- End Manual Injection ---

            if self.easyocr_reader:
                print("Extracting room descriptions from original image...")
                self.extract_room_descriptions(cv_original_image) 
            else:
                print("EasyOCR not available. Skipping text extraction.")

            
            
            print(f"Processed: {len(self.room_dimensions)} rooms, {len(self.walls)} walls, {len(self.curved_walls)} curves. Scale: {self.scale_factor:.2f} px/ft")
            
            
        except Exception as e:
            print("Processing Error", f"Error processing image: {str(e)}")
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()



    def detect_walls(self, image_cv):
        if image_cv is None:
            print("Error: Received None image in detect_walls")
            return []

        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        inverted_gray = cv2.bitwise_not(gray) 
        
        # TUNABLE: For black lines on white background, try 100-150. 
        # For gray lines on lighter gray, might need lower.
        binarization_threshold_val = 120 
        _, binarized = cv2.threshold(inverted_gray, binarization_threshold_val, 255, cv2.THRESH_BINARY)
        
        # cv2.imwrite("debug_binarized_initial.png", binarized) # For debugging

        # Morphological operations to merge double lines and close gaps
        # Adjust kernel sizes and iterations based on typical wall thickness and gap sizes in your plans
        kernel_rect_close_thick = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)) # Make lines thicker and connect
        closed_img = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel_rect_close_thick, iterations=2)

        # Optional: Erode a bit to thin lines if they became too thick, but risk disconnecting
        # kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # eroded_img = cv2.erode(closed_img, kernel_erode, iterations=1)
        # image_for_canny = eroded_img
        image_for_canny = closed_img

        # cv2.imwrite("debug_morph_processed.png", image_for_canny) # For debugging

        low_canny = 60  # Stricter Canny if morphology is good
        high_canny = 180
        edges = cv2.Canny(image_for_canny, low_canny, high_canny, apertureSize=3)
        # cv2.imwrite("debug_canny_edges.png", edges) # For debugging

        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=40,      # Number of votes
            minLineLength=25,  # Min length in pixels
            maxLineGap=15      # Max gap to join segments
        )

        if lines is None:
            print("No lines detected by HoughP.")
            return []
            
        detected_walls = []
        for line_segment in lines:
            x1, y1, x2, y2 = line_segment[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            if length < 15: continue # Filter very short lines again

            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.degrees(angle_rad)
            
            angle_tolerance_deg = 6 # Even stricter for H/V lines
            is_horizontal = (abs(angle_deg) < angle_tolerance_deg or 
                             abs(abs(angle_deg) - 180.0) < angle_tolerance_deg)
            is_vertical = (abs(abs(angle_deg) - 90.0) < angle_tolerance_deg)
                               
            if is_horizontal: wall_type = "horizontal"
            elif is_vertical: wall_type = "vertical"
            else: continue 
            
            detected_walls.append({
                "start": (int(x1), int(y1)), "end": (int(x2), int(y2)), 
                "type": wall_type, "length": length, "openings": [] 
            })
        
        print(f"Detected {len(detected_walls)} wall candidates after HoughP and filtering.")
        return detected_walls

    def detect_curved_walls(self, image_cv):
        if image_cv is None: return []
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        inverted_gray = cv2.bitwise_not(gray)
        binarization_threshold_val = 120 
        _, binarized = cv2.threshold(inverted_gray, binarization_threshold_val, 255, cv2.THRESH_BINARY)

        closing_kernel_size = 5
        kernel_closing = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
        morph_closed = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel_closing, iterations=2)

        low_threshold_canny = 60
        high_threshold_canny = 180
        edges = cv2.Canny(morph_closed, low_threshold_canny, high_threshold_canny)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # EXTERNAL for outer contours
        curved_walls_detected = []
        
        min_contour_length_pixels = 40 
        min_points_for_curve_approx = 4 # Need at least 4 for a decent curve segment
        
        for contour in contours:
            length = cv2.arcLength(contour, True) # Closed contour length
            if length < min_contour_length_pixels: continue
            
            # Filter out very small area contours that might be noise
            area = cv2.contourArea(contour)
            if area < 50 : # pixels^2
                continue

            epsilon_factor = 0.01 
            epsilon = epsilon_factor * length 
            approx = cv2.approxPolyDP(contour, epsilon, closed=True) # Use closed=True for contours
            
            # Check if approximation is not just a few points forming a simple polygon (like a small rectangle)
            # This is a heuristic to prefer more "curvy" shapes over small, blocky noise
            if len(approx) >= min_points_for_curve_approx:
                # Could add more checks here, e.g., aspect ratio of bounding box if it's a simple poly
                curved_walls_detected.append({
                    "points": [tuple(p[0]) for p in approx], "length": length, "openings": [] })
        return curved_walls_detected

    def extract_room_descriptions(self, image_input_cv): 
        if not self.easyocr_reader:
            print("EasyOCR reader not initialized. Skipping text extraction.")
            return
        try:
            easyocr_results = self.easyocr_reader.readtext(image_input_cv, detail=1, paragraph=False)
        except Exception as e:
            print(f"EasyOCR error in extract_room_descriptions: {e}")
            return

        all_text_detections = []
        for (bbox, text, prob) in easyocr_results:
            if prob < 0.4: continue 
            points = np.array(bbox, dtype=np.int32)
            all_text_detections.append({
                "text": text, 
                "center_x_px": np.mean(points[:, 0]), "center_y_px": np.mean(points[:, 1]),
                "min_x_px": np.min(points[:, 0]), "max_x_px": np.max(points[:, 0]),
                "min_y_px": np.min(points[:, 1]), "max_y_px": np.max(points[:, 1]),
                "bbox_pixels": points.tolist() 
            })

        processed_detection_indices = set() 

        temp_room_dimensions = self.room_dimensions.copy() 
        for room_name, room_data in temp_room_dimensions.items():
            if "pixel_bounds" in room_data and room_data.get("dim_str") == "To be OCR'd": 
                sel_min_x, sel_min_y, sel_max_x, sel_max_y = room_data["pixel_bounds"]
                
                best_match_ocr = None
                min_dist_to_sel_center = float('inf')
                sel_center_x = (sel_min_x + sel_max_x) / 2
                sel_center_y = (sel_min_y + sel_max_y) / 2
                best_match_idx = -1 

                for idx, detection in enumerate(all_text_detections):
                    if idx in processed_detection_indices: continue
                    
                    ocr_box_center_x, ocr_box_center_y = detection["center_x_px"], detection["center_y_px"]
                    if (sel_min_x <= ocr_box_center_x <= sel_max_x and
                        sel_min_y <= ocr_box_center_y <= sel_max_y):
                        
                        dist = math.sqrt((ocr_box_center_x - sel_center_x)**2 + (ocr_box_center_y - sel_center_y)**2)
                        if dist < min_dist_to_sel_center:
                            min_dist_to_sel_center = dist
                            best_match_ocr = detection
                            best_match_idx = idx 
                
                if best_match_ocr and best_match_idx != -1: 
                    parsed = self._parse_room_text(best_match_ocr["text"])
                    if parsed:
                        width_ft, length_ft, dim_str = parsed
                        current_scale = self.scale_factor if self.scale_factor > 0 else 1.0

                        self.room_dimensions[room_name].update({
                            "width": width_ft, "length": length_ft, "dim_str": dim_str,
                            "area": width_ft * length_ft,
                            "position": (best_match_ocr["center_x_px"] / current_scale, 
                                         best_match_ocr["center_y_px"] / current_scale),
                            "ocr_bbox_center_pixels": (best_match_ocr["center_x_px"], best_match_ocr["center_y_px"])
                        })
                        if room_name in self.room_positions:
                             self.room_positions[room_name].update({
                                "center_x": best_match_ocr["center_x_px"] / current_scale, 
                                "center_y": best_match_ocr["center_y_px"] / current_scale,
                                "min_x": (best_match_ocr["center_x_px"] / current_scale) - (width_ft / 2), 
                                "max_x": (best_match_ocr["center_x_px"] / current_scale) + (width_ft / 2), 
                                "min_y": (best_match_ocr["center_y_px"] / current_scale) - (length_ft / 2), 
                                "max_y": (best_match_ocr["center_y_px"] / current_scale) + (length_ft / 2)
                            })
                        processed_detection_indices.add(best_match_idx)
        
        unprocessed_text_detections = [
            det for i, det in enumerate(all_text_detections) if i not in processed_detection_indices
        ]

        if unprocessed_text_detections:
            positions = np.array([[d["center_x_px"], d["center_y_px"]] for d in unprocessed_text_detections])
            if len(positions) > 0:
                clustering = DBSCAN(eps=75, min_samples=1).fit(positions) 
                cluster_labels = clustering.labels_
                
                processed_clusters = set()
                for i, detection_from_unprocessed_list in enumerate(unprocessed_text_detections):
                    cluster_id = cluster_labels[i]
                    if cluster_id == -1 or cluster_id in processed_clusters: continue 

                    cluster_elements = [unprocessed_text_detections[j] for j, cid in enumerate(cluster_labels) if cid == cluster_id]
                    cluster_elements.sort(key=lambda e: (e["center_y_px"], e["center_x_px"]))
                    full_cluster_text = " ".join([elem["text"] for elem in cluster_elements])
                    
                    avg_cluster_center_x_px = np.mean([elem["center_x_px"] for elem in cluster_elements])
                    avg_cluster_center_y_px = np.mean([elem["center_y_px"] for elem in cluster_elements])
                    
                    parsed_dims = self._parse_room_text(full_cluster_text)
                    if parsed_dims:
                        width_ft, length_ft, dim_str = parsed_dims
                        room_name_from_text = "Room"; 
                        
                        name_match = re.search(r'\b(kitchen|bath(?:room)?|bed(?:room)?|living|dining|office|hallway|garage|closet|study|room|master|guest|play|nook|den|pantry|foyer|laundry)\b', full_cluster_text.lower(), re.IGNORECASE)
                        if name_match:
                            room_name_from_text = name_match.group(1).capitalize()
                        else: 
                            candidate_name = re.sub(r'[^a-zA-Z0-9\s]', '', full_cluster_text).strip()
                            candidate_name_parts = candidate_name.split()
                            if candidate_name_parts:
                                room_name_from_text = " ".join(candidate_name_parts[:2]).title() if len(candidate_name_parts) >1 else candidate_name_parts[0].title()
                            if not room_name_from_text : room_name_from_text = "Area"

                        counter = 1; final_room_name = room_name_from_text
                        while final_room_name in self.room_dimensions: 
                            counter += 1; final_room_name = f"{room_name_from_text} {counter}"
                        
                        room_type = self.determine_room_type(final_room_name)
                        current_scale = self.scale_factor if self.scale_factor > 0 else 1.0
                        pos_x_ft = avg_cluster_center_x_px / current_scale
                        pos_y_ft = avg_cluster_center_y_px / current_scale

                        self.room_dimensions[final_room_name] = {
                            "width": width_ft, "length": length_ft, "dim_str": dim_str, "area": width_ft * length_ft, 
                            "type": room_type, "position": (pos_x_ft, pos_y_ft), 
                            "ocr_bbox_center_pixels": (avg_cluster_center_x_px, avg_cluster_center_y_px) 
                        }
                        self.room_positions[final_room_name] = {
                            "center_x": pos_x_ft, "center_y": pos_y_ft,
                            "min_x": pos_x_ft - width_ft / 2, "max_x": pos_x_ft + width_ft / 2, 
                            "min_y": pos_y_ft - length_ft / 2, "max_y": pos_y_ft + length_ft / 2 
                        }
                    processed_clusters.add(cluster_id)

    def _parse_room_text(self, text): 
        dim_pattern = re.compile(
            r"(\d+)(?:['\‘\’`]\s*(?:(\d{1,2})\s*[\"”])?)?"  
            r"\s*[xX]\s*"                                   
            r"(\d+)(?:['\‘\’`]\s*(?:(\d{1,2})\s*[\"”])?)?"  
        )
        processed_text = text.replace("’", "'").replace("”", '"').replace("`","'")
        
        match = dim_pattern.search(processed_text)
        if match:
            try:
                w_ft_str, w_in_str, l_ft_str, l_in_str = match.groups()
                
                width_ft = float(w_ft_str)
                if w_in_str: width_ft += float(w_in_str) / 12.0
                
                length_ft = float(l_ft_str)
                if l_in_str: length_ft += float(l_in_str) / 12.0

                has_foot_marker_w_in_match = "'" in (w_ft_str or "") or (match.group(0) and "'" in match.group(0).split('x')[0].split('X')[0])
                has_foot_marker_l_in_match = "'" in (l_ft_str or "") or (match.group(0) and len(match.group(0).split('x')) > 1 and "'" in match.group(0).split('x')[1].split('X')[0]) \
                                        or (match.group(0) and len(match.group(0).split('X')) > 1 and "'" in match.group(0).split('X')[1].split('x')[0])


                if not (has_foot_marker_w_in_match or has_foot_marker_l_in_match):
                    if not w_in_str and width_ft > 40 and (width_ft % 1 == 0):
                         potential_w_from_inches = width_ft / 12.0
                         if 2 < potential_w_from_inches < 30: 
                             width_ft = potential_w_from_inches
                    if not l_in_str and length_ft > 40 and (length_ft % 1 == 0):
                         potential_l_from_inches = length_ft / 12.0
                         if 2 < potential_l_from_inches < 30:
                             length_ft = potential_l_from_inches
                                
                dim_str = f"{width_ft:.1f}' x {length_ft:.1f}'"
                return width_ft, length_ft, dim_str
            except ValueError: 
                return None
        return None

    def create_wall_segment_3d(self, plotter, p1_2d_ft, p2_2d_ft, z_start_ft, height_ft, thickness_ft, wall_color):
        dx = p2_2d_ft[0] - p1_2d_ft[0]
        dy = p2_2d_ft[1] - p1_2d_ft[1]
        length_sq = dx*dx + dy*dy
        if length_sq < 1e-6: return 
        length = math.sqrt(length_sq)

        dx_norm, dy_norm = dx/length, dy/length
        perp_x, perp_y = -dy_norm, dx_norm 

        half_thick = thickness_ft / 2.0
        
        v = [
            (p1_2d_ft[0] - perp_x*half_thick, p1_2d_ft[1] - perp_y*half_thick, z_start_ft),
            (p1_2d_ft[0] + perp_x*half_thick, p1_2d_ft[1] + perp_y*half_thick, z_start_ft),
            (p2_2d_ft[0] + perp_x*half_thick, p2_2d_ft[1] + perp_y*half_thick, z_start_ft),
            (p2_2d_ft[0] - perp_x*half_thick, p2_2d_ft[1] - perp_y*half_thick, z_start_ft),
            (p1_2d_ft[0] - perp_x*half_thick, p1_2d_ft[1] - perp_y*half_thick, z_start_ft + height_ft),
            (p1_2d_ft[0] + perp_x*half_thick, p1_2d_ft[1] + perp_y*half_thick, z_start_ft + height_ft),
            (p2_2d_ft[0] + perp_x*half_thick, p2_2d_ft[1] + perp_y*half_thick, z_start_ft + height_ft),
            (p2_2d_ft[0] - perp_x*half_thick, p2_2d_ft[1] - perp_y*half_thick, z_start_ft + height_ft)
        ]
        vertices = np.array(v)
        
        faces = np.array([
            4, 0, 1, 2, 3,  
            4, 7, 6, 5, 4,  
            4, 0, 4, 7, 3,  
            4, 1, 5, 6, 2,  
            4, 0, 1, 5, 4,  
            4, 3, 2, 6, 7   
        ]).ravel()


        segment_mesh = pv.PolyData(vertices, faces)
        plotter.add_mesh(segment_mesh, color=wall_color, smooth_shading=False,name="Layer_Walls")

    def create_wall_with_openings(self, plotter, wall_data_px, overall_height_ft, thickness_ft, scale_factor):
        start_px_orig = np.array(wall_data_px["start"])
        end_px_orig = np.array(wall_data_px["end"])
        wall_length_px_orig = wall_data_px["length"]
        
        if wall_length_px_orig < 1e-6 or scale_factor < 1e-6: return

        wall_unit_vec_px = (end_px_orig - start_px_orig) / wall_length_px_orig if wall_length_px_orig > 1e-9 else np.array([0,0])
        wall_color = self.materials["wall"]
        
        processed_openings = []
        for op_px_data in wall_data_px.get("openings", []):
            center_pos_on_wall_px = op_px_data["position_on_wall"]
            width_px = op_px_data["width_px"]
            
            op_start_dist_px = center_pos_on_wall_px - width_px / 2.0
            op_end_dist_px = center_pos_on_wall_px + width_px / 2.0
            
            op_start_dist_px = max(0, op_start_dist_px)
            op_end_dist_px = min(wall_length_px_orig, op_end_dist_px)
            if op_end_dist_px <= op_start_dist_px + 1e-3: continue 

            processed_openings.append({
                "start_dist_px": op_start_dist_px, 
                "end_dist_px": op_end_dist_px,
                "width_px": op_end_dist_px - op_start_dist_px, 
                "height_px": op_px_data["height_px"],
                "sill_px": op_px_data["sill_px"], 
                "type": op_px_data["type"]
            })
        processed_openings.sort(key=lambda o: o["start_dist_px"])

        current_wall_pos_px = 0.0 

        for op in processed_openings:
            op_s_px = op["start_dist_px"] 
            op_e_px = op["end_dist_px"]   
            
            op_width_ft = op["width_px"] / scale_factor
            op_height_ft = op["height_px"] / scale_factor
            op_sill_ft = op["sill_px"] / scale_factor

            if op_s_px > current_wall_pos_px + 1e-3: 
                seg_start_pt_px = start_px_orig + wall_unit_vec_px * current_wall_pos_px
                seg_end_pt_px = start_px_orig + wall_unit_vec_px * op_s_px
                self.create_wall_segment_3d(plotter, seg_start_pt_px / scale_factor, seg_end_pt_px / scale_factor, 
                                            0, overall_height_ft, thickness_ft, wall_color)

            op_seg_start_pt_px = start_px_orig + wall_unit_vec_px * op_s_px
            op_seg_end_pt_px   = start_px_orig + wall_unit_vec_px * op_e_px
            
            op_seg_start_ft = op_seg_start_pt_px / scale_factor
            op_seg_end_ft   = op_seg_end_pt_px / scale_factor

            if op_sill_ft > 1e-3: 
                self.create_wall_segment_3d(plotter, op_seg_start_ft, op_seg_end_ft,
                                            0, op_sill_ft, thickness_ft, wall_color)

            header_start_z_ft = op_sill_ft + op_height_ft
            if header_start_z_ft < overall_height_ft - 1e-3: 
                header_height_ft = overall_height_ft - header_start_z_ft
                self.create_wall_segment_3d(plotter, op_seg_start_ft, op_seg_end_ft,
                                            header_start_z_ft, header_height_ft, thickness_ft, wall_color)
            
            op_center_pt_px = start_px_orig + wall_unit_vec_px * (op_s_px + op["width_px"] / 2.0)
            op_center_pt_ft = op_center_pt_px / scale_factor
            
            wall_angle_rad = math.atan2(wall_unit_vec_px[1], wall_unit_vec_px[0])

            if op["type"] == "door":
                self.create_door_model(plotter, op_center_pt_ft, op_width_ft, op_height_ft, thickness_ft, wall_angle_rad)
            elif op["type"] == "window":
                self.create_window_model(plotter, op_center_pt_ft, op_width_ft, op_height_ft, op_sill_ft, thickness_ft, wall_angle_rad)

            current_wall_pos_px = op_e_px 

        if current_wall_pos_px < wall_length_px_orig - 1e-3 : 
            seg_start_pt_px = start_px_orig + wall_unit_vec_px * current_wall_pos_px
            self.create_wall_segment_3d(plotter, seg_start_pt_px / scale_factor, end_px_orig / scale_factor, 
                                        0, overall_height_ft, thickness_ft, wall_color)

    def create_door_model(self, plotter, center_pos_2d_ft, width_ft, height_ft, wall_thickness_ft, angle_rad_wall):
        panel_color = self.materials["door_panel"]
        door_panel_thickness = 0.15 

        door_panel = pv.Cube(center=(0, 0, height_ft / 2), 
                             x_length=width_ft, 
                             y_length=door_panel_thickness, 
                             z_length=height_ft)
        door_panel.rotate_z(math.degrees(angle_rad_wall), inplace=True)
        door_panel.translate(list(center_pos_2d_ft) + [0], inplace=True) # center_pos_2d_ft is already a tuple/list
        plotter.add_mesh(door_panel, color=panel_color, smooth_shading=False,name="Layer_Doors")


    def create_window_model(self, plotter, center_pos_2d_ft, width_ft, height_ft, sill_ft, wall_thickness_ft, angle_rad_wall):
        glass_color = self.materials["window_glass"]
        glass_thickness = 0.1 

        glass_pane = pv.Cube(center=(0, 0, sill_ft + height_ft / 2), 
                             x_length=width_ft, 
                             y_length=glass_thickness, 
                             z_length=height_ft)
        glass_pane.rotate_z(math.degrees(angle_rad_wall), inplace=True)
        glass_pane.translate(list(center_pos_2d_ft) + [0], inplace=True)
        plotter.add_mesh(glass_pane, color=glass_color, opacity=0.6, smooth_shading=False,name="Layer_Windows")
        
    def create_curved_wall(self, plotter, points_2d_ft, height_ft, thickness_ft):
        if len(points_2d_ft) < 2: return
        
        path_points = np.array([(p[0], p[1], 0.0) for p in points_2d_ft]) 
        num_path_points = len(path_points)
        half_thickness = thickness_ft / 2.0
        
        offset_vectors_3d = [] 
        for i in range(num_path_points):
            tangent_3d = np.zeros(3)
            if i == 0: 
                if num_path_points > 1: tangent_3d = path_points[i+1] - path_points[i]
            elif i == num_path_points - 1: 
                tangent_3d = path_points[i] - path_points[i-1]
            else: 
                tangent_prev = path_points[i] - path_points[i-1]
                tangent_next = path_points[i+1] - path_points[i]
                norm_prev = np.linalg.norm(tangent_prev); norm_next = np.linalg.norm(tangent_next)
                if norm_prev > 1e-9: tangent_prev /= norm_prev
                if norm_next > 1e-9: tangent_next /= norm_next
                tangent_3d = (tangent_prev + tangent_next) 
                norm_avg_tangent = np.linalg.norm(tangent_3d)
                if norm_avg_tangent > 1e-9 : tangent_3d /= norm_avg_tangent

            tangent_2d = tangent_3d[:2] 
            norm_tangent_2d = np.linalg.norm(tangent_2d)
            
            normal_vec_2d = np.array([0.0, 1.0]) 
            if norm_tangent_2d > 1e-9:
                normalized_tangent_2d = tangent_2d / norm_tangent_2d
                normal_vec_2d = np.array([-normalized_tangent_2d[1], normalized_tangent_2d[0]])
            elif offset_vectors_3d: 
                prev_offset_dir = offset_vectors_3d[-1][:2]
                prev_offset_norm = np.linalg.norm(prev_offset_dir)
                if prev_offset_norm > 1e-9:
                    normal_vec_2d = prev_offset_dir / prev_offset_norm
            elif num_path_points > 1 and i + 1 < num_path_points : 
                fallback_tangent = path_points[i+1] - path_points[i]
                fallback_tangent_2d = fallback_tangent[:2]
                norm_fallback_tangent_2d = np.linalg.norm(fallback_tangent_2d)
                if norm_fallback_tangent_2d > 1e-9:
                    normalized_fallback_tangent_2d = fallback_tangent_2d / norm_fallback_tangent_2d
                    normal_vec_2d = np.array([-normalized_fallback_tangent_2d[1], normalized_fallback_tangent_2d[0]])

            offset_vectors_3d.append(np.array([normal_vec_2d[0], normal_vec_2d[1], 0.0]) * half_thickness)

        points_side1_base = path_points - np.array(offset_vectors_3d)
        points_side2_base = path_points + np.array(offset_vectors_3d)
        
        all_vertices_list = []
        for p1, p2 in zip(points_side1_base, points_side2_base):
            all_vertices_list.extend([p1, p2])
        
        num_base_vertices_total_strip = len(all_vertices_list) 
        
        for i in range(num_base_vertices_total_strip):
            p_base = all_vertices_list[i]
            all_vertices_list.append((p_base[0], p_base[1], height_ft))
            
        vertices_np = np.array(all_vertices_list)
        
        faces_list = []
        n_segments = num_path_points - 1 
        if n_segments <= 0: return

        offset_to_top_vertices = num_path_points * 2 
        
        for i in range(n_segments):
            idx_b_s1_curr = i * 2; idx_b_s2_curr = i * 2 + 1    
            idx_b_s1_next = (i + 1) * 2; idx_b_s2_next = (i + 1) * 2 + 1
            
            idx_t_s1_curr = idx_b_s1_curr + offset_to_top_vertices
            idx_t_s2_curr = idx_b_s2_curr + offset_to_top_vertices
            idx_t_s1_next = idx_b_s1_next + offset_to_top_vertices
            idx_t_s2_next = idx_b_s2_next + offset_to_top_vertices

            faces_list.append([4, idx_b_s1_curr, idx_b_s1_next, idx_t_s1_next, idx_t_s1_curr]) 
            faces_list.append([4, idx_b_s2_next, idx_b_s2_curr, idx_t_s2_curr, idx_t_s2_next]) 
            faces_list.append([4, idx_t_s1_curr, idx_t_s2_curr, idx_t_s2_next, idx_t_s1_next]) 

        if num_path_points > 0: 
            idx_b_s1_start = 0; idx_b_s2_start = 1   
            idx_t_s1_start = idx_b_s1_start + offset_to_top_vertices
            idx_t_s2_start = idx_b_s2_start + offset_to_top_vertices
            faces_list.append([4, idx_b_s2_start, idx_b_s1_start, idx_t_s1_start, idx_t_s2_start]) 
            
            idx_b_s1_end = (num_path_points - 1) * 2
            idx_b_s2_end = idx_b_s1_end + 1
            idx_t_s1_end = idx_b_s1_end + offset_to_top_vertices
            idx_t_s2_end = idx_b_s2_end + offset_to_top_vertices
            faces_list.append([4, idx_b_s1_end, idx_b_s2_end, idx_t_s2_end, idx_t_s1_end]) 

        if not faces_list: return
        try:
            curved_wall_mesh = pv.PolyData(vertices_np, faces=np.hstack(faces_list))
            if curved_wall_mesh.n_points > 0 and curved_wall_mesh.n_cells > 0:
                 plotter.add_mesh(
                  curved_wall_mesh,
                  color=self.materials["wall"],
                  smooth_shading=False,
                   name="Layer_Walls"
                 )
            else:
                print("Warning: Generated curved wall mesh is empty or invalid.")
        except Exception as e:
            print(f"Error creating PolyData for curved wall: {e} with {len(vertices_np)} vertices and faces_list: {len(faces_list)} cells.")

    def create_furniture(self, plotter, room_type, room_bounds_ft, room_height):
        x_min, x_max, y_min, y_max = room_bounds_ft
        width = x_max - x_min; length = y_max - y_min
        center_x, center_y = x_min + width/2, y_min + length/2

        if width <= 1e-3 or length <= 1e-3: return 
        furniture_color = self.materials["furniture"]

        if room_type == "Bedroom":
            bed_w, bed_l, bed_h = min(width * 0.6, 6.0), min(length * 0.7, 7.0), 2.0 
            if bed_w > 1.0 and bed_l > 1.5 : 
                if width < length: 
                    bed_x_pos, bed_y_pos = center_x, y_min + bed_l/2 + 0.5 
                else: 
                    bed_w, bed_l = bed_l, bed_w 
                    bed_x_pos, bed_y_pos = x_min + bed_w/2 + 0.5, center_y
                
                bed_bounds = [bed_x_pos - bed_w/2, bed_x_pos + bed_w/2,
                              bed_y_pos - bed_l/2, bed_y_pos + bed_l/2,
                              0, bed_h]
                plotter.add_mesh(pv.Box(bounds=bed_bounds), color=furniture_color["bed"],name="Layer_Furniture")

        elif room_type == "Kitchen":
            counter_h, counter_d = 2.9, 2.0 
            if length > counter_d + 0.5 : 
                plotter.add_mesh(pv.Box(bounds=[x_min, x_max, y_max - counter_d, y_max, 0, counter_h]), 
                                 color=furniture_color["counter"],name="Layer_Furniture"
) 
            if width > counter_d + 0.5 :
                plotter.add_mesh(pv.Box(bounds=[x_max - counter_d, x_max, y_min, y_max - (counter_d if length > counter_d + 0.5 else 0) , 0, counter_h]), 
                                 color=furniture_color["counter"],name="Layer_Furniture"
) 
            if width > 7 and length > 7: 
                island_w, island_l = min(width*0.3, 4), min(length*0.25, 3)
                if island_w > 1.5 and island_l > 1.5:
                    island_bounds = [center_x - island_w/2, center_x + island_w/2,
                                     center_y - island_l/2, center_y + island_l/2,
                                     0, counter_h]
                    plotter.add_mesh(pv.Box(bounds=island_bounds), color=furniture_color["island"],name="Layer_Furniture"
)

        elif room_type == "Living Room":
            sofa_max_w, sofa_max_d, sofa_h = min(width * 0.7, 7), min(length*0.35, 3.0), 2.5             
            if sofa_max_w > 2.0 and sofa_max_d > 1.5:
                sofa_actual_w, sofa_actual_d = sofa_max_w, sofa_max_d
                sofa_x_pos, sofa_y_pos = center_x, y_min + sofa_actual_d/2 + 0.5 
                
                if width > length * 1.1: 
                    sofa_actual_w = sofa_max_w; sofa_actual_d = sofa_max_d 
                    sofa_x_pos = center_x; sofa_y_pos = y_min + sofa_actual_d/2 + 0.5 
                elif length > width * 1.1: 
                    sofa_actual_w = sofa_max_d; sofa_actual_d = sofa_max_w 
                    sofa_x_pos = x_min + sofa_actual_d/2 + 0.5; sofa_y_pos = center_y
                
                sofa_bounds = [sofa_x_pos - sofa_actual_w/2, sofa_x_pos + sofa_actual_w/2,
                               sofa_y_pos - sofa_actual_d/2, sofa_y_pos + sofa_actual_d/2,
                               0, sofa_h]
                plotter.add_mesh(pv.Box(bounds=sofa_bounds), color=furniture_color["sofa"],name="Layer_Furniture"
)
    
    def generate_3d_model(self):
        if not self.room_dimensions and not self.walls and not self.curved_walls:
            print("Error", "No data to generate a model. Process an image or add rooms/walls.")
            return
        try:
            current_height_ft = self.default_height
            current_wall_thickness_ft = self.wall_thickness
            current_font_size = self.label_font_size
            show_labels_flag = self.show_labels_in_3d
        except ValueError:
           #print("Input Error", "Room height, wall thickness, and font size must be valid numbers.")
            return
        if current_height_ft <=0 or current_wall_thickness_ft <=0:
           #print("Input Error", "Height and thickness must be positive.")
            return

        plotter = pv.Plotter(window_size=[1000,800]) 
        plotter.background_color = "#202020" 
        plotter.enable_shadows()

        all_points_ft = []
        if self.scale_factor > 0:
            for wall_data in self.walls:
                s_px, e_px = np.array(wall_data["start"]), np.array(wall_data["end"])
                all_points_ft.append(s_px / self.scale_factor)
                all_points_ft.append(e_px / self.scale_factor)
            for curve_data in self.curved_walls:
                for pt_px in curve_data["points"]:
                    all_points_ft.append(np.array(pt_px) / self.scale_factor)
        
        if not all_points_ft and self.room_dimensions:
             for room_name, data in self.room_dimensions.items():
                if "position" in data and "width" in data and "length" in data:
                    cx, cy = data["position"]
                    w, l = data["width"], data["length"]
                    all_points_ft.append((cx - w/2, cy - l/2))
                    all_points_ft.append((cx + w/2, cy + l/2))
        
        if not all_points_ft: 
             plotter.add_mesh(pv.Plane(center=(0,0,-0.1), direction=(0,0,1), i_size=50, j_size=50),
                              color=self.materials["floor"], name="Layer_Floor")
        else:
            all_points_ft_np = np.array(all_points_ft)
            min_coord_x = np.min(all_points_ft_np[:,0])
            max_coord_x = np.max(all_points_ft_np[:,0])
            min_coord_y = np.min(all_points_ft_np[:,1])
            max_coord_y = np.max(all_points_ft_np[:,1])

            floor_padding = 5.0 
            floor_bounds = [min_coord_x - floor_padding, max_coord_x + floor_padding,
                            min_coord_y - floor_padding, max_coord_y + floor_padding,
                            -0.1, 0] 
            plotter.add_mesh(pv.Box(bounds=floor_bounds), color=self.materials["floor"],name="Layer_Floor")

        for room_name, data in self.room_dimensions.items():
            if "position" in data and "width" in data and "length" in data:
                width_ft = data["width"]; length_ft = data["length"]
                center_x_ft, center_y_ft = data["position"]
                if width_ft <=0 or length_ft <=0: continue
                
                if show_labels_flag: 
                    plotter.add_point_labels([(center_x_ft, center_y_ft, current_height_ft / 2)], [room_name], 
                                            font_size=current_font_size, text_color="#FFFFFF", shape=None, show_points=False,
                                            always_visible=False, point_size=10) 
            
                r_min_x = center_x_ft - width_ft / 2.0; r_max_x = center_x_ft + width_ft / 2.0
                r_min_y = center_y_ft - length_ft / 2.0; r_max_y = center_y_ft + length_ft / 2.0
                room_bounds_ft_for_furniture = (r_min_x, r_max_x, r_min_y, r_max_y)
                if width_ft * length_ft > 10: 
                    self.create_furniture(plotter, data.get("type", "Other"), room_bounds_ft_for_furniture, current_height_ft)

        for wall_data_px in self.walls: 
            self.create_wall_with_openings(plotter, wall_data_px, current_height_ft, 
                                           current_wall_thickness_ft, self.scale_factor)
        
        for curve_data_px in self.curved_walls:
            if self.scale_factor > 0:
                points_ft = [(p[0] / self.scale_factor, p[1] / self.scale_factor) for p in curve_data_px["points"]]
                self.create_curved_wall(plotter, points_ft, current_height_ft, current_wall_thickness_ft)
        
        import trimesh

        output_gltf = "output.gltf"
        output_glb = "output.glb"

        plotter.render()

        plotter.export_gltf(output_glb)

        plotter.close()

        print(f"Model exported: {output_glb}")
    

    def run(self, image_path, output_path):
        """
        Full pipeline: image → 3D model
        """

        print("🔄 Engine started")

        self.image_path = image_path
        self.original_image_pil = Image.open(image_path)

        # internal steps
        self.process_current_image()
        self.generate_3d_model()

        temp_output = "output.glb"

        if not os.path.exists(temp_output):
            raise Exception("GLB not generated")

        os.replace(temp_output, output_path)

        print("✅ Saved:", output_path)

        return output_path

if __name__ == "__main__":
    print("Use main.py FastAPI server to run this project.")