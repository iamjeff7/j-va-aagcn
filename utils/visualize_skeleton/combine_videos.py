from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

def combine_videos(parent_folder):
    print('Combining...')
    clips = []
    for i in os.listdir(parent_folder):
        clips.append(VideoFileClip(os.path.join(parent_folder, i)))

    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile('combined_video.mp4')


def remove_videos(parent_folder):
    print('Removing...')
    for idx, i in enumerate(os.listdir(parent_folder)):
        fn = os.path.join(parent_folder, i)
        os.remove(fn)
        print(f'{idx+1:2d}. [DELETED] {i}')

if __name__ == '__main__':
    parent_folder = 'to_be_combined'
    combine_videos(parent_folder)
    remove_videos(parent_folder)
    print('\n\nDone\n')
