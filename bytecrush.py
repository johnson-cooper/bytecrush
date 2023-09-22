import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import numpy as np
from tqdm import tqdm
from tkinter import PhotoImage
from PIL import Image, ImageTk
import threading
import queue
import multiprocessing
from moviepy.editor import VideoFileClip, AudioFileClip
import platform
import subprocess
import os


cv2.ocl.setUseOpenCL(True)




# Function for RealESRGAN-based upscaling
def upscale_with_realesrgan(temp_images, output_path, outscale, realesrgan_options):
    try:
        cmd = [
            "python", "inference_realesrgan.py",
            "-n", realesrgan_options["model_name"],
            "-i", temp_images,
            "-o", output_path,
            "--outscale", str(outscale),
            "--fp32"
        ]

        # Add other options
        for option, value in realesrgan_options.items():
            if option not in ["model_name"]:
                cmd.append(f"--{option}")
                if value is not None:
                    cmd.append(str(value))

        subprocess.run(cmd, check=True)

        print("RealESRGAN upscaling complete. Output saved as", output_path)

    except Exception as e:
        print("An error occurred during RealESRGAN upscaling:", str(e))
        


def upscale_and_enhance_video(input_path, output_path, temp_upscaled_images_path, scale_factor, sharpen_intensity, denoise_strength, outscale_value=2, realesrgan_options=None):
    
    try:
        if realesrgan_options is not None:
            # Apply RealESRGAN upscaling
            
            upscale_with_realesrgan(temp_upscaled_images_path, outscale_value, realesrgan_options)

            # Load the upscaled image using OpenCV
            upscaled_image = cv2.imread(temp_upscaled_images_path)

            # Initialize the video writer with the upscaled image dimensions
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (upscaled_image.shape[1], upscaled_image.shape[0]))

            # Release the temporary image
            cv2.destroyAllWindows()

        else:
            if scale_factor is None or scale_factor <= 0:
                raise ValueError("Scale factor must be specified and greater than 0 when RealESRGAN is disabled.")

            else:
            # Open the input video file
                cap = cv2.VideoCapture(input_path)

                # Get the original video's frame width and height
                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))

                # Calculate the new frame dimensions after upscaling
                new_width = int(frame_width * scale_factor)
                new_height = int(frame_height * scale_factor)

                # Define the codec for video
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, 30.0, (new_width, new_height))

                # Calculate the total number of frames in the video
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Create a tqdm progress bar
                progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

            # Loop through the frames of the input video
        while True:
            ret, frame = cap.read()

            # Break the loop if we have reached the end of the video
            if not ret:
                break

            # Apply sharpening with user-defined intensity
            if sharpen_intensity > 0:
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
                frame = cv2.filter2D(frame, -1, kernel)

            # Apply deinterlacing with user-defined strength
            
            
            # Resize the frame to the new dimensions
            resized_frame = cv2.resize(frame, (new_width, new_height))

            # Write the resized frame to the output video
            out.write(resized_frame)

            # Update the progress bar
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()

        # Release video objects
        cap.release()
        out.release()

        print("Video processing complete. Temporary video saved as", output_path)

    except Exception as e:
        print("An error occurred:", str(e))



from tqdm import tqdm

def create_images_from_video(input_video_path, output_image_folder):
    try:
        # Open the input video file
        cap = cv2.VideoCapture(input_video_path)
        frame_number = 0

        # Calculate the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a tqdm progress bar
        progress_bar = tqdm(total=total_frames, desc="Creating Images", unit="frame")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Save the frame as an image in the output image folder
            image_filename = f"frame_{frame_number:04d}.png"
            image_path = os.path.join(output_image_folder, image_filename)
            cv2.imwrite(image_path, frame)

            frame_number += 1

            # Update the progress bar
            progress_bar.update(1)

        cap.release()
        progress_bar.close()

        print("Images created from video frames. Images saved in", output_image_folder)

    except Exception as e:
        print("An error occurred during image creation:", str(e))

# Example usage:

output_image_folder = "temp_images"  # Replace with the folder where you want to save the images

# Create the image folder if it doesn't exist
os.makedirs(output_image_folder, exist_ok=True)







def upscale_button_click():
    input_video_path = input_path_var.get()
    create_images_from_video(input_video_path, output_image_folder)
    temp_video_path = "temp_video.mp4"  # Temporary video file
    temp_compiledvideo_path="temp2.mp4"
    output_video_path = output_path_var.get()  # Use the specified output path
    scale_factor_str = scale_factor_entry.get()  # Get the scale factor as a string
    sharpen_intensity = sharpen_intensity_scale.get()
    denoise_strength = denoise_strength_scale.get()

    

    # Check if RealESRGAN upscaling is enabled
    use_realesrgan = realesrgan_checkbox.get()

    # Check if the "Use Multithreading" checkbox is selected
    use_multithreading = multithreading_checkbox.get()

    # Determine the number of threads based on the user's choice
    num_threads = multiprocessing.cpu_count() if use_multithreading else 1

    # Define RealESRGAN options
    realesrgan_options = None
    if use_realesrgan:
        realesrgan_options = {
            "model_name": "realesr-general-x4v3",  # You can change the model name as needed
            "suffix": "out",
            "ext": "auto"
        }

    # Define scale_factor outside the if-else block with a default value of 1
    scale_factor = 1

    try:
        if not use_realesrgan:
            # If RealESRGAN is not enabled, parse the scale factor
            scale_factor = float(scale_factor_str)  # Convert the string to a float

        # Get the selected outscale value from the dropdown menu
        outscale_value = "2"

        if use_realesrgan:
            # Use a temporary path for the RealESRGAN upscaled video
            temp_upscaled_images_path = "temp_upscaled_images"
            os.makedirs(temp_upscaled_images_path, exist_ok=True)
            upscale_with_realesrgan(output_image_folder, temp_upscaled_images_path, outscale_value, realesrgan_options)

            # Compile the upscaled images back into a video using OpenCV
            compile_images_to_video(temp_upscaled_images_path, temp_compiledvideo_path)
            add_audio_to_video(input_video_path, temp_compiledvideo_path, output_video_path)

        if os.path.exists(temp_compiledvideo_path):
            os.remove(temp_compiledvideo_path)
            print("Temporary compiled video deleted:", temp_compiledvideo_path)
        else:
            
            upscale_and_enhance_video(input_video_path, output_video_path, temp_video_path, scale_factor, sharpen_intensity, denoise_strength)

        # Add audio to the upscaled video and save it to the final output path
            add_audio_to_video(input_video_path, temp_video_path, output_video_path)
        
            

    except ValueError as ve:
        print("ValueError:", str(ve))
    except Exception as e:
        print("An error occurred:", str(e))

          # Clean up: Remove the temporary images
    clean_temp_images(output_image_folder)

# Function to clean up temporary images in the given folder
def clean_temp_images(folder_path):
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("Temporary images in", folder_path, "have been deleted.")
    except Exception as e:
        print("An error occurred while cleaning up temporary images:", str(e))

def compile_images_to_video(temp_upscaled_images_path, temp_compiledvideo_path):
    try:
        # Get a list of image file names in the directory
        image_files = sorted([os.path.join(temp_upscaled_images_path, img) for img in os.listdir(temp_upscaled_images_path) if img.endswith(('.jpg', '.jpeg', '.png'))])

        if not image_files:
            raise Exception("No upscaled images found in the directory.")

        # Read the first image to get dimensions
        first_image = cv2.imread(image_files[0])
        height, width, layers = first_image.shape

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_compiledvideo_path, fourcc, 30.0, (width, height))

        # Create a tqdm progress bar
        progress_bar = tqdm(total=len(image_files), desc="Compiling Video", unit="frame")

        # Write the images to the video
        for image_file in image_files:
            frame = cv2.imread(image_file)
            out.write(frame)
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()

        # Release the video writer
        out.release()

        # Clean up: Remove the temporary images
        for image_file in image_files:
            os.remove(image_file)

        print("Images compiled into a video. Output saved as", temp_compiledvideo_path)

    except Exception as e:
        print("An error occurred during image compilation:", str(e))

     

# New function for multithreaded video processing
def upscale_and_enhance_video_multithreaded(input_path, output_path, scale_factor, sharpen_intensity, denoise_strength, num_threads):
    try:
        # Open the input video file
        cap = cv2.VideoCapture(input_path)

        # Get the original video's frame width and height
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        # Calculate the new frame dimensions after upscaling
        new_width = int(frame_width * scale_factor)
        new_height = int(frame_height * scale_factor)

        # Define the codec and create a VideoWriter object to save the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (new_width, new_height))

        # Calculate the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a tqdm progress bar
        progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

        # Create queues for passing frames between threads
        input_queue = queue.Queue()
        output_queue = queue.Queue()

        # Define a function for frame processing in a worker thread
        def process_frame():
            while True:
                frame = input_queue.get()
                if frame is None:
                    break

                # Apply sharpening with user-defined intensity
                if sharpen_intensity > 0:
                    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
                    frame = cv2.filter2D(frame, -1, kernel)

            

              
                # Resize the frame to the new dimensions
                resized_frame = cv2.resize(frame, (new_width, new_height))

                # Put the processed frame into the output queue
                output_queue.put(resized_frame)

                # Update the progress bar
                progress_bar.update(1)

        # Create worker threads for frame processing
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=process_frame)
            thread.daemon = True
            thread.start()
            threads.append(thread)

        # Loop through the frames of the input video and put them into the input queue
        while True:
            ret, frame = cap.read()

            # Break the loop if we have reached the end of the video
            if not ret:
                break

            input_queue.put(frame)

        # Signal worker threads to finish
        for _ in range(num_threads):
            input_queue.put(None)

        # Wait for all worker threads to finish processing
        for thread in threads:
            thread.join()

        # Write frames from the output queue to the output video
        while not output_queue.empty():
            resized_frame = output_queue.get()
            out.write(resized_frame)

        # Close the progress bar
        progress_bar.close()

        # Release video objects
        cap.release()
        out.release()

        print("Video upscaling and enhancement complete. Output saved as", output_path)

    except Exception as e:
        print("An error occurred:", str(e))


def add_audio_to_video(input_video_path, temp_video_path, output_video_path):
    try:
        # Load the processed video without audio using moviepy
        video_clip = VideoFileClip(temp_video_path)

        # Load the original audio from the input video using moviepy
        audio_clip = AudioFileClip(input_video_path)

        # Set the audio of the video clip to the loaded audio clip
        video_clip = video_clip.set_audio(audio_clip)

        # Write the final video with audio
        video_clip.write_videofile(output_video_path, codec='libx264')

        print("Audio added to the video. Output saved as", output_video_path)

    except Exception as e:
        print("An error occurred:", str(e))

def update_preview():
    try:
        cap = cv2.VideoCapture(input_path_var.get())
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Apply the selected effects to the frame (same as in upscale_and_enhance_video)
            frame = cv2.resize(frame, (400, 300))
            if sharpen_intensity_scale.get() > 0:
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
                frame = cv2.filter2D(frame, -1, kernel)
            if denoise_strength_scale.get() > 0:
                frame = cv2.fastNlMeansDenoisingColored(frame, None, denoise_strength_scale.get(), 10, 7, 21)
           
            # Convert the frame to RGB for displaying in Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            preview_label.config(image=photo)
            preview_label.photo = photo
            preview_label.update()
    except Exception as e:
        print("An error occurred during preview:", str(e))

def start_preview():
    # Start a thread for the video preview
    preview_thread = threading.Thread(target=update_preview)
    preview_thread.daemon = True
    preview_thread.start()

# Create the main GUI window
root = tk.Tk()
root.title("Bytecrush - Video Upscaler and Enhancer")
# fixed issue with platform dependency
if platform.system() == 'Windows':
    # Your Windows-specific code here
    root.iconbitmap('favicon.ico')

style = ttk.Style()

# Set the theme to "clam" or any other built-in theme
style.theme_use("clam")  # You can change "clam" to other available themes



        

try:
    # Set dark mode theme
    root.tk_setPalette(background='#FFFFFF', foreground='#1e1e1e')

    # Set larger window size
    root.geometry("800x600")

    # Load the background image
    bg_image = PhotoImage(file="background.png")  # Replace "background.png" with your image file

    # Create a Label widget to display the background image
    background_label = tk.Label(root, image=bg_image)
    background_label.place(relwidth=1, relheight=1)

    # Buttons and forms on the left
    form_frame = tk.Frame(root)
    form_frame.pack(side="left", padx=20, pady=10)

 

    # Create a Label widget for the video preview
    preview_label = tk.Label(form_frame)
    preview_label.pack(side="right", padx=20, pady=10)


   
    # Input video path File Dialog button
    input_path_label = tk.Label(form_frame, text="Select Input Video:", fg='#1e1e1e', bg='white')
    input_path_label.pack()
    input_path_var = tk.StringVar()
    input_path_entry = tk.Entry(form_frame, textvariable=input_path_var, state='readonly')
    input_path_entry.pack()

    def browse_input_path():
        file_path = filedialog.askopenfilename(title="Select Input Video File", filetypes=[("Video Files", "*.mp4")])
        if file_path:
            input_path_var.set(file_path)

    input_browse_button = tk.Button(form_frame, text="Browse", command=browse_input_path)
    input_browse_button.pack()

    # Output video path File Dialog button
    output_path_label = tk.Label(form_frame, text="Select Output Video:", fg='#1e1e1e', bg='white')
    output_path_label.pack()
    output_path_var = tk.StringVar()
    output_path_entry = tk.Entry(form_frame, textvariable=output_path_var, state='readonly')
    output_path_entry.pack()

    def browse_output_path():
        file_path = filedialog.asksaveasfilename(title="Save Output Video As", filetypes=[("Video Files", "*.mp4")])
        if file_path:
            output_path_var.set(file_path)

    output_browse_button = tk.Button(form_frame, text="Browse", command=browse_output_path)
    output_browse_button.pack()

    # Scale factor label and entry
    scale_factor_label = tk.Label(form_frame, text="Scale Factor:", fg='#1e1e1e', bg='white')
    scale_factor_label.pack()
    scale_factor_entry = tk.Entry(form_frame)
    scale_factor_entry.pack()

    # Sharpening intensity slider
    sharpen_intensity_label = tk.Label(form_frame, text="Sharpening Intensity", fg='#1e1e1e', bg='white')
    sharpen_intensity_label.pack(anchor="w")
    sharpen_intensity_scale = ttk.Scale(form_frame, from_=0, to=10, orient="horizontal")
    sharpen_intensity_scale.set(0)  # Default value
    sharpen_intensity_scale.pack(fill="x")

    # Denoise strength slider
    denoise_strength_label = tk.Label(form_frame, text="Denoise Strength", fg='#1e1e1e', bg='white')
    denoise_strength_label.pack(anchor="w")
    denoise_strength_scale = ttk.Scale(form_frame, from_=0, to=10, orient="horizontal")
    denoise_strength_scale.set(0)  # Default value
    denoise_strength_scale.pack(fill="x")

   

    # Upscale button
    upscale_button = tk.Button(form_frame, text="Upscale and Enhance Video", command=upscale_button_click)
    upscale_button.pack()

    #checkbox for esrgan
    realesrgan_checkbox = tk.BooleanVar()
    realesrgan_checkbox.set(False)  # Default to disabled
    realesrgan_checkbox_button = tk.Checkbutton(form_frame, text="Enable RealESRGAN Upscaling", variable=realesrgan_checkbox)
    realesrgan_checkbox_button.pack()


    # Create a "Use Multithreading" checkbox
    multithreading_checkbox = tk.BooleanVar()
    multithreading_checkbox.set(False)  # Default to single-threaded
    multithreading_checkbox_button = tk.Checkbutton(form_frame, text="Use Multithreading", variable=multithreading_checkbox)
    multithreading_checkbox_button.pack()

    # Create a "Preview" button
    preview_button = tk.Button(form_frame, text="Preview", command=start_preview)
    preview_button.pack()

    
    # Start the GUI main loop
    root.mainloop()

except Exception as e:
    print("An error occurred:", str(e))