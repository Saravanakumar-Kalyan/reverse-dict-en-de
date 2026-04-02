"""
Word alignment and corpus processing utilities using FastAlign.
"""

import re
import collections
from typing import Dict, List, Tuple


def tokenize(text: str) -> str:
    """
    Tokenize text by separating punctuation from words.
    
    Example: "hello." -> "hello ."
    
    Args:
        text: Input text string
        
    Returns:
        Tokenized text with separated punctuation
    """
    text = re.sub(r'([.,!?():;])', r' \1 ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def prepare_fast_align_data(src_file: str, trg_file: str, output_file: str):
    """
    Prepare parallel corpus for FastAlign alignment tool.
    
    Combines source and target files into FastAlign format:
    Source Sentence ||| Target Sentence
    
    Args:
        src_file: Path to source language file (English)
        trg_file: Path to target language file (German)
        output_file: Path to output file in FastAlign format
    """
    try:
        with open(src_file, 'r', encoding='utf-8') as src, \
             open(trg_file, 'r', encoding='utf-8') as trg, \
             open(output_file, 'w', encoding='utf-8') as out:
            
            line_count = 0
            for s_line, t_line in zip(src, trg):
                s_tok = tokenize(s_line)
                t_tok = tokenize(t_line)
                
                # Skip empty lines
                if s_tok == '' or t_tok == '':
                    continue
                
                out.write(f"{s_tok} ||| {t_tok}\n")
                line_count += 1
            
            print(f"Successfully processed {line_count} sentence pairs.")
            print(f"Output saved to: {output_file}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}. Check your file paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def clean_parallel_corpus(file1_path: str, file2_path: str, 
                         output1_path: str, output2_path: str):
    """
    Remove empty lines from parallel corpus.
    
    Ensures source and target sentences remain aligned after cleaning.
    
    Args:
        file1_path: Path to first language file (English)
        file2_path: Path to second language file (German)
        output1_path: Path to cleaned first language file
        output2_path: Path to cleaned second language file
    """
    try:
        with open(file1_path, 'r', encoding='utf-8') as f1, \
             open(file2_path, 'r', encoding='utf-8') as f2, \
             open(output1_path, 'w', encoding='utf-8') as out1, \
             open(output2_path, 'w', encoding='utf-8') as out2:
            
            removed_count = 0
            total_count = 0
            
            for line1, line2 in zip(f1, f2):
                total_count += 1
                clean1 = line1.strip()
                clean2 = line2.strip()
                
                if clean1 and clean2:
                    out1.write(clean1 + '\n')
                    out2.write(clean2 + '\n')
                else:
                    removed_count += 1
        
        print(f"Processing complete!")
        print(f"Total pairs processed: {total_count}")
        print(f"Empty pairs removed:   {removed_count}")
        print(f"Cleaned pairs saved:   {total_count - removed_count}")
    
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_alignments(alignment_file: str, source_file: str, target_file: str) -> Dict[str, List]:
    """
    Load FastAlign output and create word translation dictionary.
    
    Maps each source word to its most frequently aligned target words.
    
    Args:
        alignment_file: Path to FastAlign output file
        source_file: Path to source language sentences
        target_file: Path to target language sentences
        
    Returns:
        Dictionary mapping source words to their translation info
    """
    translation_map = collections.defaultdict(list)
    
    try:
        with open(alignment_file, encoding='utf-8') as f_a, \
             open(source_file, encoding='utf-8') as f_s, \
             open(target_file, encoding='utf-8') as f_t:
            
            for line_a, line_s, line_t in zip(f_a, f_s, f_t):
                aligns = line_a.strip().split()
                src_tokens = line_s.strip().split()
                tgt_tokens = line_t.strip().split()
                
                # Parse alignment pairs
                for pair in aligns:
                    try:
                        s_idx, t_idx = map(int, pair.split('-'))
                        if 0 <= s_idx < len(src_tokens) and 0 <= t_idx < len(tgt_tokens):
                            src_word = src_tokens[s_idx].lower()
                            tgt_word = tgt_tokens[t_idx].lower()
                            translation_map[src_word].append(tgt_word)
                    except (ValueError, IndexError):
                        continue
        
        # Get most common translation for each source word
        final_dict = {}
        for src_word, tgt_list in translation_map.items():
            counter = collections.Counter(tgt_list)
            most_common = counter.most_common(1)
            if most_common:
                final_dict[src_word] = most_common
        
        print(f"Loaded {len(final_dict)} aligned word pairs.")
        return final_dict
    
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}


def collapse_dictionary(alignment_dict: Dict) -> Dict[str, str]:
    """
    Clean and simplify alignment dictionary.
    
    Removes punctuation and flattens nested structure to simple word mapping.
    
    Args:
        alignment_dict: Raw alignment dictionary from load_alignments
        
    Returns:
        Simplified dictionary mapping English words to German words
    """
    collapsed = {}
    
    for en_word, de_data in alignment_dict.items():
        # Clean English word
        en_clean = str(en_word).lower().strip()
        for char in [',', '.', ':', ';', '!', '?']:
            en_clean = en_clean.replace(char, '')
        
        # Extract German word from nested structure
        if isinstance(de_data, (list, tuple)) and len(de_data) > 0:
            first_item = de_data[0]
            
            if isinstance(first_item, (list, tuple)):
                de_word_raw = first_item[0]
            else:
                de_word_raw = first_item
            
            # Clean German word
            de_clean = str(de_word_raw).lower().strip()
            for char in [',', '.', ':', ';', '!', '?']:
                de_clean = de_clean.replace(char, '')
            
            if en_clean and de_clean:
                collapsed[en_clean] = de_clean
    
    print(f"Collapsed dictionary: {len(collapsed)} pairs ready.")
    return collapsed
