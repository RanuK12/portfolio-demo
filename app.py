from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def read_root():
    return {'msg': 'Hola, soy el demo de portfolio!'}

