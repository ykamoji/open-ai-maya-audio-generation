import os
import glob
import shutil
import argparse

from PostProcess.process_image_colors import extract_image_scheme


def move_audio_and_subtitle_files(source_base_path: str, destination_folder: str):
    """
    Finds all .m4a and .srt files within specified subdirectories of the source path
    and moves them to the destination folder.

    Args:
        source_base_path (str): The base path where the 'audios' subdirectory is located.
                                  e.g., '/content/drive/MyDrive/AI/Outputs/'
        destination_folder (str): The target folder to move the files to.
                                  e.g., '/content/drive/MyDrive/AI/Audiobook'
    """
    os.makedirs(destination_folder, exist_ok=True)
    print(f"Ensured existence of destination folder: {destination_folder}")

    files_to_move = []
    for ext in ['m4a', 'srt']:
        # Correct glob pattern to look for .wav and .srt files in immediate subdirectories of 'audios'
        search_pattern = os.path.join(source_base_path, f"audios/*/*.{ext}")
        files_to_move.extend(glob.glob(search_pattern))

    if not files_to_move:
        print(f"No .wav or .srt files found in {os.path.join(source_base_path, 'audios/*/')}")
        return

    print(f"Found {len(files_to_move)} files to move.")
    moved_count = 0
    for f in files_to_move:
        try:
            shutil.move(f, os.path.join(destination_folder, os.path.basename(f)))
            print(f"Moved {os.path.basename(f)} to {destination_folder}")
            moved_count += 1
        except Exception as e:
            print(f"Error moving file {f}: {e}")

    print(f"Successfully moved {moved_count} out of {len(files_to_move)} files to {destination_folder}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ColorScheme")

    parser.add_argument("--s", type=str,
                        default="output", help="Audio source Paths")
    parser.add_argument("--d", type=str,
                        default="/Users/ykamoji/Documents/Audiobooks", help="Audio destination Paths")
    args = parser.parse_args()

    move_audio_and_subtitle_files(args.s, args.d)

    extract_image_scheme(args.d)




