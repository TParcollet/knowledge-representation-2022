## Execute program :

Graph prediction

    python main.py ogbg-molhiv

Node prediction 

    python main.py ogbn-arxiv --flag

## Node prediction :

GraphSage + FLAG :
- Valid: 73.32% Test: 71.70%
- Valid: 72.43% Test: 71.35%

## Graph prediction :

GIN :
- Valid: 82.75% Test: 78.05%  
- Valid: 82.45% Test: 77.18%  

GAT :
- Valid: 78.99% Test: 73.85%
- Valid: 78.80% Test: 70.66%


GEN :
- Valid: 80.33% Test: 78.67%
- Valid: 81.45% Test: 78.12%

SAGE :
- Valid: 80.04% Test: 72.50%
