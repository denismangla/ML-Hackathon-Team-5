from fastapi import FastAPI

app = FastAPI(
    title='PathSnap API',
    description='GPS trajectory map-matching backend scaffold',
    version='0.1.0',
)


@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'PathSnap', 'prediction_ready': False}


# Do not freeze /predict until the organiser's deployment PDF / evaluator schema
# is confirmed and the real candidate -> model -> HMM -> route pipeline is integrated.
