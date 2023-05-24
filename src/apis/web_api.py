from fastapi import APIRouter, HTTPException
import csv
from pathlib import Path
from src.models.message import Message
from src.models.mlmodel import MLModel
from src.models.qna import QnA
from src.models.qnamodel import QnAModel
import boto3
import tensorflow as tf
from datetime import datetime
import time

router = APIRouter()
mlModel = MLModel()
qnaModel = QnAModel()

@router.get("/")
def home():
    model_path = Path("../..", "ML", "deeplearning", "pretrained", "models", "spam_no_spam_model").resolve()
    return model_path

@router.get("/microapp1")
async def get_micro_app_1_data():
    data= []
    with open(Path("src", "data", "microapp1.csv").resolve(), encoding="utf-8") as csvf:
        csvReader = csv.DictReader(csvf)
        for row in csvReader:
            data.append(row)
    return data

@router.post("/microapp1")
async def create_micro_app_1_data(message: Message):
    path =  Path("src", "data", "microapp1.csv").resolve()
    with open(path, encoding="utf-8") as csvf:
        message.id = sum(1 for line in csvf)
    with open(path, mode='a', encoding="utf-8",newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow([message.id, message.message, "unknown", message.insertDate, "null"])
    
    sqs_resp = await send_sqs_message(message)
    try:
        print(sqs_resp['MessageId'])
        message.sqsMessageId = sqs_resp['MessageId']
    except Exception as e:
        print(e)
    time.sleep(10)
    return message

@router.get("/microapp2")
async def get_micro_app_2_data():
    data= []
    with open(Path("src", "data", "microapp2.csv").resolve(), encoding="utf-8") as csvf:
        csvReader = csv.DictReader(csvf)
        for row in csvReader:
            data.append(row)
    return data

@router.post("/microapp2")
async def create_micro_app_2_data(qna: QnA):
    path =  Path("src", "data", "microapp2.csv").resolve()
    with open(path, encoding="utf-8") as csvf:
        qna.id = sum(1 for line in csvf)
    with open(path, mode='a', encoding="utf-8",newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow([qna.id, qna.question, "unknown", qna.insertDate, "null"])
    
    sqs_resp = await send_qna_sqs_message(qna)
    try:
        print(sqs_resp['MessageId'])
        qna.sqsMessageId = sqs_resp['MessageId']
    except Exception as e:
        print(e)
    time.sleep(10)
    return qna

@router.post("/sqs")
async def send_sqs_message(message: Message):
    client = boto3.client('sqs')
    queue_url = client.get_queue_url(QueueName='bkdemomicroapponequeue')
    resp = client.send_message(
            QueueUrl=queue_url['QueueUrl'],
            MessageBody=message.message,
            MessageAttributes={
                "message_type": {"StringValue":"Micro App 1 Message Created","DataType":"String"},
                "message_id": {"StringValue":str(message.id),"DataType":"Number"},
                "message_insert_date": {"StringValue":str(message.insertDate),"DataType":"String"},
            }
    )
    return resp

@router.post("/sqsqna")
async def send_qna_sqs_message(qna: QnA):
    client = boto3.client('sqs')
    queue_url = client.get_queue_url(QueueName='bkdemomicroapptwoqueue')
    resp = client.send_message(
            QueueUrl=queue_url['QueueUrl'],
            MessageBody=qna.question,
            MessageAttributes={
                "message_type": {"StringValue":"Micro App 1 Message Created","DataType":"String"},
                "message_id": {"StringValue":str(qna.id),"DataType":"Number"},
                "message_insert_date": {"StringValue":str(qna.insertDate),"DataType":"String"},
            }
    )
    return resp

@router.put('/update')
async def update_row(message: Message):
    if message.id == None:
        raise HTTPException(status_code=404, detail="Message not found")
    
    path =  Path("src", "data", "microapp1.csv").resolve()
    op = open(path, "r")
    dt = csv.DictReader(op)
    print(dt)
    up_dt = []
    for r in dt:
        print(r['id'])
        if str(message.id) == r['id']:
            row = {'id': message.id,
                'message': message.message,
                'spamnospam': message.spamOrNot,
                'insertdatetime': message.insertDate,
                'updatedatetime': message.updateDate}
        else:
            row = {'id': r['id'],
                'message': r['message'],
                'spamnospam': r['spamnospam'],
                'insertdatetime': r['insertdatetime'],
                'updatedatetime': r['updatedatetime']}
        row['updatedatetime'] = datetime.now().strftime('%m/%d/%Y')
        up_dt.append(row)
    print(up_dt)
    op.close()
    op = open(path, "w", newline='')
    headers = ['id','message','spamnospam','insertdatetime','updatedatetime']
    data = csv.DictWriter(op, delimiter=',', fieldnames=headers)
    data.writerow(dict((heads, heads) for heads in headers))
    data.writerows(up_dt)
    
    op.close()

    return message

@router.put('/updateqna')
async def update_qna_row(qna: QnA):
    if qna.id == None:
        raise HTTPException(status_code=404, detail="Question not found")
    
    path =  Path("src", "data", "microapp2.csv").resolve()
    op = open(path, "r")
    dt = csv.DictReader(op)
    print(dt)
    up_dt = []
    for r in dt:
        print(r['id'])
        if str(qna.id) == r['id']:
            row = {'id': qna.id,
                'question': qna.question,
                'answer': qna.answer,
                'insertdatetime': qna.insertDate,
                'updatedatetime': qna.updateDate}
        else:
            row = {'id': r['id'],
                'question': r['question'],
                'answer': r['answer'],
                'insertdatetime': r['insertdatetime'],
                'updatedatetime': r['updatedatetime']}
        row['updatedatetime'] = datetime.now().strftime('%m/%d/%Y')
        up_dt.append(row)
    print(up_dt)
    op.close()
    op = open(path, "w", newline='')
    headers = ['id','question','answer','insertdatetime','updatedatetime']
    data = csv.DictWriter(op, delimiter=',', fieldnames=headers)
    data.writerow(dict((heads, heads) for heads in headers))
    data.writerows(up_dt)
    
    op.close()

    return qna

@router.post('/predict/spamnospam')
async def predict_spam_not_spam(message: Message):
    string_list = [message.message]
    encodings  = mlModel.tokenizer(string_list, max_length=512, truncation=True, padding=True)
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings)))
    print(len(dataset))
    preds = mlModel.model.predict(dataset.batch(1))
    result = tf.nn.softmax(preds.logits, axis=1).numpy()[0]
    if result[1] > result[0] and result[1] >= 0.75:
        message.spamOrNot = 'Spam'
    else:
        message.spamOrNot = 'Not Spam'
    return await update_row(message)

@router.post('/predict/qna')
async def predict_qna(qna: QnA):
    result = qnaModel.nlp({
        'question': qna.question,
        'context': qnaModel.context
    })
    qna.answer = result['answer']
    return await update_qna_row(qna)

