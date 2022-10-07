import requests
import uvicorn
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from fastapi.responses import FileResponse
from dependencies import classification
from typing import List

app = FastAPI()

class request_body_train(BaseModel):
    task_type: str
    report_id: str
    table_id: str
    bearer_token: str
    target: str


class pipeline:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route('/', self.home, methods=['GET'])
        self.router.add_api_route('/train', self.train, methods=['POST'], response_model=List[dict])
        self.router.add_api_route('/results', self.results, methods=['GET'])

    def home(self):
        return {"Hello! I am a model that predicts churn. To train me, send a POST request to /train with a target variable. To get my results, send a GET request to /results."}
    def results(self):
        # return {self.task.sorted_importances_idx}
        return FileResponse('feature_importance.png')

    def train(self, data: request_body_train):
        if data.task_type == 'classification':
            self.task = classification()

        # This is the URL and Bearer token of the API endpoint
        headers = {"Authorization": "Bearer {}".format(data.bearer_token)}
        url = "https://api.ignatius.io/api/report/export?reportId={}&tableId={}&exportType=csv&size=-1&tblName=1".format(data.report_id, data.table_id)
        response = requests.get(url, headers=headers)
        self.df = self.task.convert_to_dataframe(response)
        return self.task.train(self.df, data.target)


app.include_router(pipeline().router)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

# {
#   "task_type": "classification",
#   "report_id": "ti2coyqg1",
#   "table_id": "2363",
#   "bearer_token": "Go9L61T0-HYuNPyYhINW3BuHWIBvZbuVnvu7RAAirJ8",
#   "target": "churn"
# }