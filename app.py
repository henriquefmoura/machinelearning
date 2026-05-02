import runpy
import os

# Executa ml_insights.py como script principal (evita cache de sys.modules entre reruns)
_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_insights.py")
runpy.run_path(_script, run_name="__main__")
