import os
import re

def embed_code():
    with open('Attractive_Thesis.tex', 'r', encoding='utf-8') as f:
        content = f.read()
        
    def replacer(match):
        filepath = match.group(1)
        # Read the file content
        try:
            with open(filepath, 'r', encoding='utf-8') as pyf:
                code = pyf.read()
            return "\\begin{lstlisting}[language=Python]\n" + code + "\n\\end{lstlisting}"
        except Exception as e:
            return f"% Error loading {filepath}: {e}"
            
    # Replace \lstinputlisting[language=Python]{filepath}
    pattern = r'\\lstinputlisting\[language=Python\]\{(.*?)\}'
    new_content = re.sub(pattern, replacer, content)
    
    with open('Attractive_Thesis.tex', 'w', encoding='utf-8') as f:
        f.write(new_content)
        
    print("Successfully embedded Python code directly into Attractive_Thesis.tex!")

if __name__ == "__main__":
    embed_code()
