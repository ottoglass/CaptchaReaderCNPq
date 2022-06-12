import json
import os

from Pages import CNPqLattes

Nome = "Santa Maria"
Numero_resultados = "3"
lattes = CNPqLattes()
resultados = lattes.search(Nome, 3000, buscar_demais=False)

with open("resultados.json", "w") as json_file:
    json_resultados = json.dumps(resultados)
    json_file.write(json_resultados)
print(resultados)
