import os

def build():
    # Read the original thesis
    with open('Master_Thesis.tex', 'r') as f:
        thesis_content = f.read()
        
    # We need to insert the listings package at the top
    packages = """\\usepackage{listings}
\\usepackage{xcolor}
\\definecolor{codegreen}{rgb}{0,0.6,0}
\\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\\definecolor{codepurple}{rgb}{0.58,0,0.82}
\\definecolor{backcolour}{rgb}{0.95,0.95,0.92}
\\lstdefinestyle{mystyle}{
    backgroundcolor=\\color{backcolour},   
    commentstyle=\\color{codegreen},
    keywordstyle=\\color{magenta},
    numberstyle=\\tiny\\color{codegray},
    stringstyle=\\color{codepurple},
    basicstyle=\\ttfamily\\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2
}
\\lstset{style=mystyle}
"""
    # Insert packages before \geometry
    thesis_content = thesis_content.replace('\\geometry', packages + '\\geometry')
    
    # Read the python files
    files_to_embed = [
        ('Query Analyzer (NLP)', 'dynamic_edge/query_analyzer.py'),
        ('Dynamic Segmenter (Temporal Scaling)', 'dynamic_edge/dynamic_segmenter.py'),
        ('Audio-Visual Frame Selector (CLIP/CLAP MMR)', 'dynamic_edge/edge_frame_selection.py'),
        ('Paper Evaluator (Ablation Study)', 'dynamic_edge/paper_evaluator.py')
    ]
    
    appendix = "\\appendix\n\\chapter{Source Code Implementations}\nThis appendix contains the complete, executable Python source code for the Dynamic Edge-Cloud Collaboration Framework developed for this thesis. It includes the query analysis, dynamic segmentation, audio-triggered extraction, and the semantic marginal relevance pipelines.\\newline\n\n"
    
    for title, path in files_to_embed:
        appendix += f"\\section{{{title}}}\n"
        appendix += "\\begin{lstlisting}[language=Python]\n"
        with open(path, 'r', encoding='utf-8') as pf:
            appendix += pf.read()
        appendix += "\n\\end{lstlisting}\n\n"
        
    # Insert appendix before \end{document}
    thesis_content = thesis_content.replace('\\end{document}', appendix + '\\end{document}')
    
    with open('Final_Submission.tex', 'w', encoding='utf-8') as f:
        f.write(thesis_content)
        
    print("Successfully built Final_Submission.tex!")

if __name__ == "__main__":
    build()
