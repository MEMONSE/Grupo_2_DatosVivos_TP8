import gradio as gr
import pandas as pd
import pickle


# Definir parámetros
PARAMS_NAME = [
    "Age",
    "Class",
    "Wifi",
    "Booking",
    "Seat",
    "Checkin"
]


# Cargar el modelo
with open("C:/Users/Usuario/Desktop/datos Vivos/TP8/TP8/rf.pkl", "rb") as f:
    model = pickle.load(f)

# Cargar el nombre de las columnas
COLUMNS_PATH = "C:/Users/Usuario/Desktop/datos Vivos/TP8/TP8/categories_ohe.pickle"
with open(COLUMNS_PATH, 'rb') as handle:
    ohe_tr = pickle.load(handle)


def predict(*args):
    answer_dict = {}

    for i in range(len(PARAMS_NAME)):
        answer_dict[PARAMS_NAME[i]] = [args[i]]

    single_instance = pd.DataFrame.from_dict(answer_dict)
    
    # Reformat columns
    single_instance_ohe = pd.get_dummies(single_instance).reindex(columns = ohe_tr).fillna(0)
    
    prediction = model.predict(single_instance_ohe)

    response = format(prediction[0], '.2f')
    print(response)
    return ("Satisfecho 😀" if response == "1.00" else "Insatisfecho o indiferente 😕")


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Satisfacción de aerólineas ✈️
        """
    )

    with gr.Row():
        with gr.Column():

            gr.Markdown(
                """
                ## ¿Cliente satisfecho?
                """
            )
            Age = gr.Slider(label="Edad", minimum=5, maximum=99, step=1, randomize=True)

            Class = gr.Radio(
                label="Clase",
                choices=['Business', 'Eco', 'Eco Plus'],
                value='Eco Plus',
                )
            
            
            Wifi = gr.Slider(label="Servicio de wifi", minimum=0, maximum=5, step=1, randomize=True)

            Booking = gr.Slider(label="Facilidad de registro", minimum=0, maximum=5, step=1, randomize=True)

            Seat = gr.Dropdown(
                label="Comodidad del asiento",
                choices=[1,2,3,4,5],
                multiselect=False,
                value=3,
                )

            Checkin = gr.Dropdown(
                label="Experiencia con el Checkin",
                choices=[1,2,3,4,5],
                multiselect=False,
                value=3,
                )
            
        with gr.Column():

            gr.Markdown(
                """
                ## Predicción
                """
            )

            label = gr.Label(label="¿Cómo se fue el cliente?")
            predict_btn = gr.Button(value="Evaluar")
            predict_btn.click(
                predict, #función que se ejecuta al apretar Evaluar
                inputs=[
                   Age,
                   Class,
                   Wifi,
                   Booking,
                   Seat,
                   Checkin,
 
                ], #lista de input que el usuario completa
                outputs=[label], #restultado de la función 
                api_name="¿Cómo se fue el cliente?"
            )
            
            
            # Imagen
            image = gr.Image(
                value="C:/Users/Usuario/Desktop/datos Vivos/TP8/TP8/FOTO.jpg",
                label="Gracias por tu evaluación ✈️",
                show_label=True,
                interactive=False
            )

demo.launch(share = True)