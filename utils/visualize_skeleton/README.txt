# To visualize
    1. Need skeleton and label
    2. File type = .npy
    3. Shape = (batch_size, 300, 150) 
        - batch size must be at least 1
    4. Edit file name in the code
    5. Run >> python3 visualize_skeleton.py
    6. File will be saved in "skeleton_videos" folder


# To combine videos
    1. Move the videos to "to_be_combined" folder
    2. The videos will be removed after combinding by default
       , but you can edit it in the code
    3. Run >> python3 combine_videos.py
    4. Output filename is "combined_videos.mp4"
