import logging
import os
import re
from typing import List, Sequence, Tuple, Union

import os.path as op

# Configure logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# File Collection
# -----------------------------------------------------------------------------

def remove_extension(filenames: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Remove up to two file extensions from the given filename(s).

    For example, 'image.nii.gz' or 'model.pt.tar' becomes 'image' or 'model'
    if the inner extension is among known double extensions (`.nii`, `.pt`).

    Parameters
    ----------
    filenames : str or list of str
        Filename or list of filenames from which to remove extensions.

    Returns
    -------
    str or list of str
        The filename(s) without extension. If a string was provided, a string is returned.
        If a list was provided, a list is returned.

    Notes
    -----
    - Only `.nii` and `.pt` are considered for nested/double extensions.
    - The outermost extension is always removed; the inner one only if it is in the known list.
    """
    known_double_extensions = ['.nii', '.pt']
    return_single = False

    if isinstance(filenames, str):
        filenames = [filenames]
        return_single = True

    cleaned_names = []
    for name in filenames:
        name_no_extension, _ = os.path.splitext(name)
        name_no_extension_2, extension_2 = os.path.splitext(name_no_extension)

        if extension_2 in known_double_extensions:
            cleaned_names.append(name_no_extension_2)
        else:
            cleaned_names.append(name_no_extension)

    return cleaned_names[0] if return_single else cleaned_names

def get_parent_path(
    paths: Union[str, List[str]],
    level: int = 1,
    remove_ext: bool = False
) -> Union[Tuple[str, str], Tuple[List[str], List[str]]]:
    """
    Extract parent directory and file/directory name from a path at a given depth level.

    This function returns the parent path and the name of a file or directory, taken at the
    specified `level` from the end of the path. If `level` is negative, the `level` last components
    are concatenated with underscores (`_`) to form the name.

    Parameters
    ----------
    paths : str or list of str
        One or more file or directory paths.

    level : int, optional
        Depth level from the end of the path to extract the name. 
        - If positive, extracts a single component at that depth.
        - If negative, concatenates the last `abs(level)` components. Default is 1.

    remove_ext : bool, optional
        If True, remove the extension from the last component before extracting. Default is False.

    Returns
    -------
    tuple
        A tuple of (parent_path, name), where:
        - If input was a single string, both are strings.
        - If input was a list, both are lists of strings.
    """
    return_single = False

    if isinstance(paths, str):
        paths = [paths]
        return_single = True

    parent_paths: List[str] = []
    names: List[str] = []

    concat = False
    if level < 0:
        level = -level
        concat = True

    for path in paths:
        path = path.rstrip('/')
        components = path.split('/')
        depth = len(components)

        if remove_ext:
            components[-1] = remove_extension(components[-1])

        if concat:
            name = '_'.join(components[depth - level:])
        else:
            name = components[depth - level]

        parent_path = '/'.join(components[:depth - level])

        parent_paths.append(parent_path)
        names.append(name)

    if return_single:
        return parent_paths[0], names[0]
    else:
        return parent_paths, names


def get_files(
    directories: Union[str, Sequence[str]],
    patterns: Union[str, Sequence[str]],
    max_items: int = -1,
    flatten: bool = True,
) -> List[str]:
    """
    Recursively collect files matching final pattern within nested regex-filtered directories.
    """
    dirs = ([directories] if isinstance(directories, str) else list(directories))
    pats = ([patterns] if isinstance(patterns, str) else list(patterns))

    # Traverse intermediate patterns
    for pat in pats[:-1]:
        rx = re.compile(pat)

        new_dirs = []
        for d in dirs:
            if not op.isdir(d):
                continue
            new_dirs += [op.join(d, f) for f in os.listdir(d) if rx.search(f)]
        dirs = new_dirs

    # Final file match
    final_rx = re.compile(pats[-1])

    collected = []
    for d in dirs:
        if not op.isdir(d):
            continue

        matches = []
        for f in sorted(os.listdir(d)):
            if final_rx.search(f):
                path = op.join(d, f)
                if flatten:
                    collected.append(path)
                else:
                    matches.append(path)

        if not flatten:
            collected.append(matches)

        if max_items > 0 and len(matches) != max_items:

            logger.warning(f"Directory {d} has {len(matches)} items, expected {max_items}")
    return collected


def get_directories(
    directories: Union[str, List[str]],
    regex: Union[str, List[str]],
) -> List[str]:
    """
    Recursively find subdirectories matching a sequence of regular expressions.

    This function searches through a list of directories, filtering subdirectories
    at each level using a corresponding regular expression from `regex`. The search is 
    performed recursively: each level of `regex` corresponds to one level of subdirectories,
    with the last pattern optionally matching filenames (though this function only returns directories).

    Parameters
    ----------
    directories : str or list of str
        A directory path or a list of paths in which to begin the recursive search.
    regex : str or list of str
        A regular expression or list of regular expressions used to match subdirectories
        at each level of recursion.

    Returns
    -------
    list of str
        A list of absolute paths to directories that match the full sequence of regular expressions.
        The list is sorted alphabetically at each directory level.

    Notes
    -----
    - If a given directory does not exist, it is silently skipped.
    - Only directories are returned; files are ignored even if they match the regex.
    - This function uses `os.listdir` for sorting and filtering, which does not perform natural sorting.
    """
    if isinstance(directories, str):
        directories = [directories]
    if not directories:
        print(" ** NO directories found!!")
        return []

    if isinstance(regex, str):
        regex = [regex]

    if len(regex) == 1:
        matched_dirs: List[str] = []
        pattern = re.compile(regex[0])

        for base_dir in directories:
            abs_dir = op.abspath(base_dir)
            if not op.isdir(abs_dir):
                continue

            sub_items = sorted(os.listdir(abs_dir))
            for item in sub_items:
                item_path = op.join(abs_dir, item)
                if op.isdir(item_path) and pattern.search(item):
                    matched_dirs.append(item_path)

        return matched_dirs
    else:
        current_dirs = directories
        for current_regex in regex:
            current_dirs = get_directories(current_dirs, current_regex)

        return current_dirs
