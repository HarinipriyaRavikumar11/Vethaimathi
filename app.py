from flask import Flask, render_template, request, send_file, session, redirect
import asyncio
import edge_tts
import requests
import os
import uuid
from PIL import Image
from werkzeug.utils import secure_filename
from transformers import pipeline
from offline_tips import OFFLINE_TIPS, OFFLINE_TIPS_ENGLISH


app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "vethaimathi-local-secret-key"
)


# =========================================================
# AI DISEASE DETECTION MODEL
# =========================================================

MODEL_NAME = "Kathir56/plant-disease-tamilnadu"

print("Loading AI disease detection model...")

classifier = pipeline(
    "image-classification",
    model=MODEL_NAME
)

print("AI model loaded successfully.")


# =========================================================
# UPLOAD SETTINGS
# =========================================================

UPLOAD_FOLDER = os.path.join(
    "static",
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}


def allowed_file(filename):
    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================================================
# DISEASE INFORMATION
# Tamil + English
# =========================================================

DISEASE_INFO = {

    "Pepper,_bell___Bacterial_spot": {

        "tamil_name": "மிளகாய் பாக்டீரியா புள்ளி நோய்",
        "english_name": "Pepper Bacterial Spot",

        "crop_ta": "மிளகாய்",
        "crop_en": "Pepper",

        "symptoms_ta":
            "இலைகளில் சிறிய கருமையான புள்ளிகள் "
            "அல்லது காயங்கள் தோன்றலாம். "
            "பாதிக்கப்பட்ட பகுதிகள் படிப்படியாக அதிகரிக்கலாம்.",

        "symptoms_en":
            "Small dark spots or lesions may appear on the leaves. "
            "The affected areas may gradually increase.",

        "cause_ta":
            "பாக்டீரியா தொற்று காரணமாக இந்த நோய் ஏற்படலாம். "
            "அதிக ஈரப்பதம் மற்றும் ஈரமான சூழல் நோய் பரவுவதற்கு உதவலாம்.",

        "cause_en":
            "This disease may be caused by a bacterial infection. "
            "High humidity and wet conditions can support disease spread.",

        "management_ta":
            "பாதிக்கப்பட்ட இலைகளை அகற்றவும். "
            "வயலில் நல்ல காற்றோட்டத்தை பராமரிக்கவும். "
            "இலைகளில் அதிக ஈரப்பதம் தேங்காமல் பார்த்துக்கொள்ளவும்.",

        "management_en":
            "Remove affected leaves. Maintain good air circulation "
            "in the field and avoid excessive moisture on the leaves."
    },


    "Pepper,_bell___healthy": {

        "tamil_name": "மிளகாய் இலை ஆரோக்கியமாக உள்ளது",
        "english_name": "Healthy Pepper Leaf",

        "crop_ta": "மிளகாய்",
        "crop_en": "Pepper",

        "symptoms_ta":
            "குறிப்பிடத்தக்க நோய் அறிகுறிகள் காணப்படவில்லை.",

        "symptoms_en":
            "No significant disease symptoms were detected.",

        "cause_ta":
            "தற்போதைய AI கணிப்பின்படி இலை ஆரோக்கியமாக இருக்கலாம்.",

        "cause_en":
            "According to the current AI prediction, the leaf may be healthy.",

        "management_ta":
            "வழக்கமான நீர்ப்பாசனம் மற்றும் பயிர் பராமரிப்பை தொடரவும். "
            "நோய் அறிகுறிகள் தோன்றுகிறதா என்பதை தொடர்ந்து கண்காணிக்கவும்.",

        "management_en":
            "Continue regular irrigation and crop care. "
            "Monitor the plant for any new disease symptoms."
    },


    "Potato___Early_blight": {

        "tamil_name": "உருளைக்கிழங்கு இலை கருகல் நோய்",
        "english_name": "Potato Early Blight",

        "crop_ta": "உருளைக்கிழங்கு",
        "crop_en": "Potato",

        "symptoms_ta":
            "இலைகளில் கருமையான புள்ளிகள் மற்றும் வளைய வடிவ "
            "பாதிப்புகள் தோன்றலாம். பாதிக்கப்பட்ட இலைகள் படிப்படியாக வாடலாம்.",

        "symptoms_en":
            "Dark spots and ring-shaped lesions may appear on leaves. "
            "Affected leaves may gradually wilt.",

        "cause_ta":
            "பூஞ்சை தொற்று காரணமாக இந்த நோய் ஏற்படலாம். "
            "ஈரப்பதமான சூழல் நோய் வளர்ச்சிக்கு சாதகமாக இருக்கலாம்.",

        "cause_en":
            "The disease may be caused by a fungal infection. "
            "Humid conditions can support disease development.",

        "management_ta":
            "பாதிக்கப்பட்ட இலைகளை அகற்றவும். வயலில் நல்ல காற்றோட்டத்தை "
            "பராமரிக்கவும். தேவையான பயிர் பாதுகாப்பு நடவடிக்கைகளுக்கு "
            "வேளாண்மை நிபுணரின் ஆலோசனையைப் பெறவும்.",

        "management_en":
            "Remove affected leaves and maintain good air circulation. "
            "Consult an agricultural expert for appropriate crop protection measures."
    },


    "Potato___Late_blight": {

        "tamil_name": "உருளைக்கிழங்கு தாமத இலை கருகல் நோய்",
        "english_name": "Potato Late Blight",

        "crop_ta": "உருளைக்கிழங்கு",
        "crop_en": "Potato",

        "symptoms_ta":
            "இலைகளில் கருமையான அல்லது பழுப்பு நிற பாதிப்புகள் தோன்றலாம். "
            "ஈரமான சூழலில் நோய் வேகமாக பரவக்கூடும்.",

        "symptoms_en":
            "Dark or brown lesions may appear on leaves. "
            "The disease can spread rapidly under wet conditions.",

        "cause_ta":
            "பூஞ்சை போன்ற நோய்க்கிருமி காரணமாக இந்த நோய் ஏற்படலாம். "
            "அதிக ஈரப்பதம் நோய் பரவலுக்கு சாதகமாக இருக்கலாம்.",

        "cause_en":
            "The disease may be caused by a fungus-like pathogen. "
            "High humidity can favor disease spread.",

        "management_ta":
            "பாதிக்கப்பட்ட பகுதிகளை கண்காணித்து அகற்றவும். "
            "வயலில் நல்ல காற்றோட்டத்தை பராமரிக்கவும். "
            "நிபுணர் ஆலோசனைப்படி பயிர் பாதுகாப்பு நடவடிக்கைகளை மேற்கொள்ளவும்.",

        "management_en":
            "Monitor and remove affected areas. Maintain good air circulation "
            "and follow crop protection advice from an agricultural expert."
    },


    "Potato___healthy": {

        "tamil_name": "உருளைக்கிழங்கு இலை ஆரோக்கியமாக உள்ளது",
        "english_name": "Healthy Potato Leaf",

        "crop_ta": "உருளைக்கிழங்கு",
        "crop_en": "Potato",

        "symptoms_ta":
            "குறிப்பிடத்தக்க நோய் அறிகுறிகள் காணப்படவில்லை.",

        "symptoms_en":
            "No significant disease symptoms were detected.",

        "cause_ta":
            "தற்போதைய AI கணிப்பின்படி இலை ஆரோக்கியமாக இருக்கலாம்.",

        "cause_en":
            "According to the current AI prediction, the leaf may be healthy.",

        "management_ta":
            "வழக்கமான பயிர் பராமரிப்பை தொடரவும். இலைகளில் புதிய "
            "புள்ளிகள் அல்லது நிறமாற்றங்கள் தோன்றுகிறதா என்பதை கண்காணிக்கவும்.",

        "management_en":
            "Continue regular crop care. Monitor the leaves for new spots "
            "or changes in color."
    },


    "Tomato___Bacterial_spot": {

        "tamil_name": "தக்காளி பாக்டீரியா புள்ளி நோய்",
        "english_name": "Tomato Bacterial Spot",

        "crop_ta": "தக்காளி",
        "crop_en": "Tomato",

        "symptoms_ta":
            "இலைகளில் சிறிய கருமையான புள்ளிகள் தோன்றலாம். "
            "பாதிக்கப்பட்ட பகுதிகள் படிப்படியாக அதிகரிக்கலாம்.",

        "symptoms_en":
            "Small dark spots may appear on leaves. "
            "The affected areas may gradually increase.",

        "cause_ta":
            "பாக்டீரியா தொற்று காரணமாக ஏற்படலாம். "
            "ஈரமான சூழல் நோய் பரவுவதற்கு உதவக்கூடும்.",

        "cause_en":
            "The disease may be caused by a bacterial infection. "
            "Wet conditions can support disease spread.",

        "management_ta":
            "பாதிக்கப்பட்ட இலைகளை அகற்றவும். "
            "வயலில் நல்ல காற்றோட்டத்தை பராமரிக்கவும்.",

        "management_en":
            "Remove affected leaves and maintain good air circulation "
            "in the field."
    },


    "Tomato___Early_blight": {

        "tamil_name": "தக்காளி ஆரம்ப இலை கருகல் நோய்",
        "english_name": "Tomato Early Blight",

        "crop_ta": "தக்காளி",
        "crop_en": "Tomato",

        "symptoms_ta":
            "இலைகளில் கருமையான புள்ளிகள் மற்றும் வளைய வடிவ "
            "பாதிப்புகள் தோன்றலாம்.",

        "symptoms_en":
            "Dark spots and ring-shaped lesions may appear on leaves.",

        "cause_ta":
            "பூஞ்சை தொற்று காரணமாக ஏற்படலாம்.",

        "cause_en":
            "The disease may be caused by a fungal infection.",

        "management_ta":
            "பாதிக்கப்பட்ட இலைகளை அகற்றி, வயலில் நல்ல காற்றோட்டத்தை பராமரிக்கவும்.",

        "management_en":
            "Remove affected leaves and maintain good air circulation."
    },


    "Tomato___Late_blight": {

        "tamil_name": "தக்காளி தாமத இலை கருகல் நோய்",
        "english_name": "Tomato Late Blight",

        "crop_ta": "தக்காளி",
        "crop_en": "Tomato",

        "symptoms_ta":
            "இலைகளில் கருமையான அல்லது பழுப்பு நிற பாதிப்புகள் தோன்றலாம்.",

        "symptoms_en":
            "Dark or brown lesions may appear on leaves.",

        "cause_ta":
            "ஈரமான மற்றும் குளிர்ச்சியான சூழலில் நோய் வேகமாக பரவக்கூடும்.",

        "cause_en":
            "The disease can spread quickly in wet and cool conditions.",

        "management_ta":
            "பாதிக்கப்பட்ட இலைகளை அகற்றவும். "
            "இலைகளில் நீர் தேங்காமல் பார்த்துக்கொள்ளவும்.",

        "management_en":
            "Remove affected leaves and avoid excessive water remaining on the leaves."
    },


    "Tomato___Septoria_leaf_spot": {

        "tamil_name": "தக்காளி இலை புள்ளி நோய்",
        "english_name": "Tomato Septoria Leaf Spot",

        "crop_ta": "தக்காளி",
        "crop_en": "Tomato",

        "symptoms_ta":
            "இலைகளில் சிறிய கரும்புள்ளிகள் தோன்றலாம். பின்னர் பாதிக்கப்பட்ட "
            "பகுதிகள் அதிகரித்து இலைகள் வாடலாம்.",

        "symptoms_en":
            "Small dark spots may appear on leaves. "
            "Affected areas may increase and leaves may eventually wilt.",

        "cause_ta":
            "பூஞ்சை தொற்று காரணமாக ஏற்படலாம். இலைகளில் அதிக ஈரப்பதம் "
            "இருப்பது நோய் பரவுவதற்கு உதவலாம்.",

        "cause_en":
            "The disease may be caused by a fungal infection. "
            "High moisture on leaves can support disease spread.",

        "management_ta":
            "பாதிக்கப்பட்ட இலைகளை அகற்றவும். இலைகளில் நீர் தேங்காமல் "
            "பார்த்துக்கொள்ளவும். வயலில் நல்ல காற்றோட்டத்தை பராமரிக்கவும்.",

        "management_en":
            "Remove affected leaves, avoid water remaining on the leaves, "
            "and maintain good air circulation."
    },


    "Tomato___Leaf_Mold": {

        "tamil_name": "தக்காளி இலை பூஞ்சை நோய்",
        "english_name": "Tomato Leaf Mold",

        "crop_ta": "தக்காளி",
        "crop_en": "Tomato",

        "symptoms_ta":
            "இலைகளில் மஞ்சள் அல்லது பழுப்பு நிற பகுதிகள் தோன்றலாம்.",

        "symptoms_en":
            "Yellow or brown areas may appear on the leaves.",

        "cause_ta":
            "அதிக ஈரப்பதம் மற்றும் குறைந்த காற்றோட்டம் "
            "நோய் வளர்ச்சிக்கு உதவக்கூடும்.",

        "cause_en":
            "High humidity and poor air circulation can support disease development.",

        "management_ta":
            "வயலில் நல்ல காற்றோட்டத்தை பராமரிக்கவும். "
            "பாதிக்கப்பட்ட இலைகளை கண்காணித்து அகற்றவும்.",

        "management_en":
            "Maintain good air circulation and monitor and remove affected leaves."
    },


    "Tomato___healthy": {

        "tamil_name": "தக்காளி இலை ஆரோக்கியமாக உள்ளது",
        "english_name": "Healthy Tomato Leaf",

        "crop_ta": "தக்காளி",
        "crop_en": "Tomato",

        "symptoms_ta":
            "குறிப்பிடத்தக்க நோய் அறிகுறிகள் காணப்படவில்லை.",

        "symptoms_en":
            "No significant disease symptoms were detected.",

        "cause_ta":
            "தற்போதைய AI கணிப்பின்படி இலை ஆரோக்கியமாக இருக்கலாம்.",

        "cause_en":
            "According to the current AI prediction, the leaf may be healthy.",

        "management_ta":
            "வழக்கமான பயிர் பராமரிப்பை தொடரவும். புதிய நோய் அறிகுறிகள் "
            "தோன்றுகிறதா என்பதை தொடர்ந்து கண்காணிக்கவும்.",

        "management_en":
            "Continue regular crop care and monitor for any new disease symptoms."
    }
}


# =========================================================
# DISEASE INFORMATION HELPER
# =========================================================

def get_disease_info(disease_name, language="ta"):

    info = DISEASE_INFO.get(disease_name)

    if info:

        if language == "en":

            return {
                "tamil_name": info["english_name"],
                "crop": info["crop_en"],
                "symptoms": info["symptoms_en"],
                "cause": info["cause_en"],
                "management": info["management_en"]
            }

        return {
            "tamil_name": info["tamil_name"],
            "crop": info["crop_ta"],
            "symptoms": info["symptoms_ta"],
            "cause": info["cause_ta"],
            "management": info["management_ta"]
        }


    clean_name = (
        disease_name
        .replace("___", " - ")
        .replace("_", " ")
        .replace(",", "")
    )

    if language == "en":

        return {
            "tamil_name": clean_name,
            "crop": "Crop not confirmed",
            "symptoms":
                "Detailed symptoms for this prediction "
                "are not available in the current information set.",
            "cause":
                "The cause cannot be confirmed using AI prediction alone.",
            "management":
                "Please test again with a clear leaf image and "
                "consult an agricultural expert if needed."
        }

    return {
        "tamil_name": clean_name,
        "crop": "பயிர் விவரம் உறுதிப்படுத்தப்படவில்லை",
        "symptoms":
            "இந்த AI கணிப்புக்கான அறிகுறிகள் தற்போது உள்ள "
            "தகவல் தொகுப்பில் கிடைக்கவில்லை.",
        "cause":
            "AI கணிப்பு மட்டும் கொண்டு நோய்க்கான காரணத்தை "
            "உறுதிப்படுத்த முடியாது.",
        "management":
            "தெளிவான இலை புகைப்படத்துடன் மீண்டும் பரிசோதிக்கவும். "
            "தேவையானால் வேளாண்மை நிபுணரிடம் உறுதிப்படுத்திக்கொள்ளவும்."
    }


# =========================================================
# IMAGE VALIDATION
# =========================================================

def validate_image(image_path):

    try:

        image = Image.open(image_path)
        image.verify()

        return True

    except Exception as e:

        print("Image validation error:", e)

        return False


# =========================================================
# AI PREDICTION
# =========================================================

def predict_disease(image_path):

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        results = classifier(
            image,
            top_k=5
        )

        if not results:
            return None, 0, []


        predictions = []

        for item in results:

            label = item["label"]

            score = round(
                float(item["score"]) * 100,
                2
            )

            info = get_disease_info(
                label,
                "ta"
            )

            predictions.append({

                "label": label,

                "tamil_name":
                    info["tamil_name"],

                "score": score
            })


        best_prediction = predictions[0]

        best_label = best_prediction["label"]

        best_score = best_prediction["score"]


        if len(predictions) >= 2:

            second_score = predictions[1]["score"]

            prediction_gap = (
                best_score - second_score
            )

        else:

            prediction_gap = 100


        MIN_CONFIDENCE = 45

        MIN_GAP = 8


        confident_prediction = (
            best_score >= MIN_CONFIDENCE
            and
            prediction_gap >= MIN_GAP
        )


        if not confident_prediction:

            return {
                "label": "Unknown",
                "score": best_score,
                "gap": prediction_gap
            }, best_score, predictions


        return {
            "label": best_label,
            "score": best_score,
            "gap": prediction_gap
        }, best_score, predictions


    except Exception as e:

        print(
            "AI Prediction Error:",
            e
        )

        raise e


# =========================================================
# LANGUAGE PAGE
# =========================================================

@app.route("/")
def language():

    language = request.args.get(
        "lang",
        ""
    )

    if language in ("ta", "en"):

        session["language"] = language

    return render_template(
        "language.html"
    )


# =========================================================
# CROP DISEASE DETECTION
# =========================================================

@app.route(
    "/disease",
    methods=["GET", "POST"]
)
def disease():

    language = session.get(
        "language",
        "ta"
    )

    if language not in ("ta", "en"):

        language = "ta"


    disease_name = None
    disease_info = None
    confidence = None
    predictions = []
    image_path = None
    error = None
    is_unknown = False


    if request.method == "POST":

        file = (
            request.files.get("leaf_image")
            or request.files.get("file")
            or request.files.get("image")
        )


        if not file or file.filename == "":

            if language == "en":

                error = (
                    "Please select a crop leaf image."
                )

            else:

                error = (
                    "ஒரு பயிரின் இலை புகைப்படத்தை "
                    "தேர்வு செய்யவும்."
                )


        elif not allowed_file(
            file.filename
        ):

            if language == "en":

                error = (
                    "Please upload only JPG, JPEG, "
                    "PNG or WEBP images."
                )

            else:

                error = (
                    "JPG, JPEG, PNG அல்லது WEBP "
                    "படத்தை மட்டும் பதிவேற்றவும்."
                )


        else:

            try:

                original_name = secure_filename(
                    file.filename
                )

                extension = (
                    original_name
                    .rsplit(".", 1)[1]
                    .lower()
                )

                unique_name = (
                    uuid.uuid4().hex
                    + "."
                    + extension
                )

                filepath = os.path.join(
                    UPLOAD_FOLDER,
                    unique_name
                )


                file.save(filepath)


                if not validate_image(filepath):

                    if language == "en":

                        error = (
                            "The image could not be read. "
                            "Please upload a clear JPG, PNG "
                            "or WEBP image."
                        )

                    else:

                        error = (
                            "படத்தை சரியாக படிக்க முடியவில்லை. "
                            "வேறு ஒரு தெளிவான JPG, PNG அல்லது WEBP "
                            "படத்தை பதிவேற்றவும்."
                        )


                else:

                    result, confidence, predictions = (
                        predict_disease(filepath)
                    )


                    if result is None:

                        if language == "en":

                            error = (
                                "AI prediction was not available. "
                                "Please upload another clear "
                                "crop leaf image."
                            )

                        else:

                            error = (
                                "AI prediction கிடைக்கவில்லை. "
                                "வேறு ஒரு தெளிவான இலை புகைப்படத்தை "
                                "பதிவேற்றவும்."
                            )


                    else:

                        disease_name = result["label"]

                        confidence = result["score"]


                        # ---------------------------------
                        # CHANGE PREDICTION DISPLAY LANGUAGE
                        # ---------------------------------

                        if language == "en":

                            for prediction in predictions:

                                info = get_disease_info(
                                    prediction["label"],
                                    "en"
                                )

                                prediction["tamil_name"] = (
                                    info["tamil_name"]
                                )


                        # ---------------------------------
                        # UNKNOWN
                        # ---------------------------------

                        if disease_name == "Unknown":

                            is_unknown = True


                            if language == "en":

                                disease_name = (
                                    "No reliable prediction"
                                )

                                disease_info = {

                                    "tamil_name":
                                        "The image could not be "
                                        "reliably identified.",

                                    "crop":
                                        "Not confirmed",

                                    "symptoms":
                                        "The AI model could not "
                                        "identify the leaf disease "
                                        "with sufficient confidence.",

                                    "cause":
                                        "The confidence difference "
                                        "between the top predictions "
                                        "was not sufficient.",

                                    "management":
                                        "Please upload a clear leaf "
                                        "image again. Make sure the "
                                        "image has good lighting."
                                }


                            else:

                                disease_name = (
                                    "நம்பகமான கணிப்பு கிடைக்கவில்லை"
                                )

                                disease_info = {

                                    "tamil_name":
                                        "இந்த படத்தை நம்பகமாக "
                                        "அடையாளம் காண முடியவில்லை",

                                    "crop":
                                        "உறுதிப்படுத்தப்படவில்லை",

                                    "symptoms":
                                        "படத்தில் உள்ள இலை நோயை "
                                        "இந்த AI model போதுமான "
                                        "நம்பகத்தன்மையுடன் "
                                        "கண்டறியவில்லை.",

                                    "cause":
                                        "Top predictions-க்கு இடையிலான "
                                        "நம்பகத்தன்மை போதுமானதாக இல்லை.",

                                    "management":
                                        "தெளிவான இலை புகைப்படத்தை "
                                        "மீண்டும் பதிவேற்றவும். "
                                        "படம் நன்றாக வெளிச்சத்துடன் "
                                        "இருப்பதை உறுதி செய்யவும்."
                                }


                        else:

                            disease_info = get_disease_info(
                                disease_name,
                                language
                            )


                        image_path = (
                            "/static/uploads/"
                            + unique_name
                        )


                        print(
                            "--------------------------------"
                        )

                        print(
                            "CROP DISEASE DETECTION"
                        )

                        print(
                            "Language       :",
                            language
                        )

                        print(
                            "Original Image :",
                            original_name
                        )

                        print(
                            "Saved Image    :",
                            unique_name
                        )

                        print(
                            "Prediction     :",
                            disease_name
                        )

                        print(
                            "Confidence     :",
                            confidence,
                            "%"
                        )

                        print(
                            "Top Predictions:",
                            predictions
                        )

                        print(
                            "--------------------------------"
                        )


            except Exception as e:

                print(
                    "Disease Detection Error:",
                    e
                )


                if language == "en":

                    error = (
                        "The image could not be analyzed. "
                        "Please upload a clear crop leaf image "
                        "and try again."
                    )

                else:

                    error = (
                        "படத்தை பகுப்பாய்வு செய்ய முடியவில்லை. "
                        "தெளிவான இலை புகைப்படத்தை மீண்டும் "
                        "பதிவேற்றவும்."
                    )


    return render_template(

        "disease.html",

        disease_name=disease_name,

        disease_info=disease_info,

        confidence=confidence,

        predictions=predictions,

        image_path=image_path,

        error=error,

        is_unknown=is_unknown,

        language=language
    )


# =========================================================
# FARMER NAME ENTRY
# =========================================================

@app.route(
    "/name",
    methods=["GET", "POST"]
)
def farmer_name():

    if request.method == "GET":

        language = request.args.get(
            "lang",
            session.get("language", "ta")
        )

        if language not in ("ta", "en"):

            language = "ta"

        session["language"] = language

        return render_template(
            "name.html",
            language=language
        )


    farmer_name = request.form.get(
        "farmer_name",
        ""
    ).strip()


    language = request.form.get(
        "language",
        session.get("language", "ta")
    )


    if language not in ("ta", "en"):

        language = "ta"


    if farmer_name:

        session["farmer_name"] = farmer_name

        session["language"] = language

        return redirect("/dashboard")


    return render_template(
        "name.html",
        language=language
    )


# =========================================================
# WEATHER FUNCTION
# =========================================================

def get_weather(region):

    region_coordinates = {

        "cuddalore": {
            "latitude": 11.7480,
            "longitude": 79.7714
        },

        "villupuram": {
            "latitude": 11.9401,
            "longitude": 79.4861
        },

        "chennai": {
            "latitude": 13.0827,
            "longitude": 80.2707
        },

        "pondicherry": {
            "latitude": 11.9416,
            "longitude": 79.8083
        }
    }


    coordinates = region_coordinates.get(

        region,

        {
            "latitude": 11.7753,
            "longitude": 79.7613
        }
    )


    url = (
        "https://api.open-meteo.com/v1/forecast"
    )


    params = {

        "latitude":
            coordinates["latitude"],

        "longitude":
            coordinates["longitude"],

        "current":
            (
                "temperature_2m,"
                "relative_humidity_2m,"
                "precipitation,"
                "wind_speed_10m"
            ),

        "daily":
            (
                "temperature_2m_max,"
                "temperature_2m_min,"
                "precipitation_probability_max,"
                "precipitation_sum"
            ),

        "forecast_days": 7,

        "timezone": "auto"
    }


    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()


        return {

            "current":
                data["current"],

            "daily":
                data["daily"],

            "available": True
        }


    except Exception as e:

        print(
            "Weather API Error:",
            e
        )


        return {

            "current": {

                "temperature_2m": 0,

                "relative_humidity_2m": 0,

                "precipitation": 0,

                "wind_speed_10m": 0
            },

            "daily": {

                "temperature_2m_max": [0],

                "temperature_2m_min": [0],

                "precipitation_probability_max": [0],

                "precipitation_sum": [0]
            },

            "available": False
        }


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    latitude = 11.7753

    longitude = 79.7613


    url = (
        "https://api.open-meteo.com/v1/forecast"
    )


    params = {

        "latitude": latitude,

        "longitude": longitude,

        "current":
            (
                "temperature_2m,"
                "relative_humidity_2m,"
                "precipitation,"
                "wind_speed_10m"
            ),

        "daily":
            (
                "temperature_2m_max,"
                "temperature_2m_min,"
                "precipitation_probability_max,"
                "precipitation_sum"
            ),

        "forecast_days": 7,

        "timezone": "auto"
    }


    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        weather = data["current"]

        daily = data["daily"]


    except Exception as e:

        print(
            "Dashboard Weather Error:",
            e
        )


        weather = {

            "temperature_2m": 0,

            "relative_humidity_2m": 0,

            "precipitation": 0,

            "wind_speed_10m": 0
        }


        daily = {

            "temperature_2m_max": [0],

            "temperature_2m_min": [0],

            "precipitation_probability_max": [0],

            "precipitation_sum": [0]
        }


    alerts = []


    current_temperature = (
        weather["temperature_2m"]
    )


    rain_probability = (
        daily[
            "precipitation_probability_max"
        ][0]
    )


    if current_temperature >= 35:

        alerts.append({

            "type": "heat",

            "title":
                "☀️ வெப்பநிலை எச்சரிக்கை",

            "message":
                (
                    "வெப்பநிலை அதிகமாக உள்ளது. "
                    "பயிர்களுக்கு போதுமான "
                    "நீர்ப்பாசனத்தை திட்டமிடுங்கள்."
                )
        })


    if (
        current_temperature >= 30
        and
        rain_probability < 40
    ):

        alerts.append({

            "type": "irrigation",

            "title":
                "💧 நீர்ப்பாசன அறிவுரை",

            "message":
                (
                    "வெப்பநிலை அதிகமாகவும் "
                    "மழை வாய்ப்பு குறைவாகவும் உள்ளது. "
                    "பயிர்களுக்கு தேவையான "
                    "நீர்ப்பாசனத்தை வழங்கவும்."
                )
        })


    elif rain_probability >= 60:

        alerts.append({

            "type": "irrigation_warning",

            "title":
                "🌧️ நீர்ப்பாசன எச்சரிக்கை",

            "message":
                (
                    "மழை வாய்ப்பு அதிகமாக உள்ளது. "
                    "தேவையற்ற நீர்ப்பாசனத்தை "
                    "தவிர்க்கவும்."
                )
        })


    farmer_name = session.get(
        "farmer_name",
        ""
    )

    language = session.get(
        "language",
        "ta"
    )


    return render_template(

        "dashboard.html",

        weather=weather,

        daily=daily,

        alerts=alerts,

        farmer_name=farmer_name,

        language=language
    )


# =========================================================
# WEATHER BASED CROP SCORE
# =========================================================

def get_weather_score(
    crop,
    temperature,
    rain_probability
):

    score = 0

    reason = ""


    if crop == "Rice":

        if 25 <= temperature <= 35:

            score += 40

            reason = (
                "Warm temperature is suitable for rice."
            )


        if rain_probability >= 50:

            score += 40

            reason += (
                " Good rainfall probability "
                "supports water requirements."
            )

        elif rain_probability >= 30:

            score += 25

            reason += (
                " Moderate rainfall can "
                "support growth."
            )

        else:

            score += 10

            reason += (
                " Low rainfall means "
                "irrigation may be required."
            )


    elif crop == "Groundnut":

        if 25 <= temperature <= 35:

            score += 40

            reason = (
                "Warm temperature is suitable for groundnut."
            )


        if 20 <= rain_probability <= 60:

            score += 35

            reason += (
                " Moderate rainfall is favorable."
            )

        elif rain_probability < 20:

            score += 15

            reason += (
                " Low rainfall may require irrigation."
            )

        else:

            score += 10

            reason += (
                " High rainfall may increase excess moisture risk."
            )


    elif crop == "Millet":

        if 25 <= temperature <= 35:

            score += 40

            reason = (
                "Warm weather is favorable for millet."
            )


        if rain_probability <= 50:

            score += 35

            reason += (
                " Moderate or low rainfall suits millet."
            )

        else:

            score += 15

            reason += (
                " Higher rainfall may reduce suitability."
            )


    elif crop == "Black Gram":

        if 25 <= temperature <= 32:

            score += 40

            reason = (
                "Suitable temperature supports black gram."
            )


        if 20 <= rain_probability <= 60:

            score += 35

            reason += (
                " Moderate rainfall is favorable."
            )

        else:

            score += 15

            reason += (
                " Current rainfall conditions are less ideal."
            )


    elif crop == "Sugarcane":

        if 25 <= temperature <= 35:

            score += 40

            reason = (
                "Warm conditions are suitable for sugarcane."
            )


        if rain_probability >= 40:

            score += 35

            reason += (
                " Good rainfall supports water requirements."
            )

        else:

            score += 15

            reason += (
                " Lower rainfall may require additional irrigation."
            )


    elif crop == "Cotton":

        if 25 <= temperature <= 35:

            score += 40

            reason = (
                "Warm temperature is suitable for cotton."
            )


        if 20 <= rain_probability <= 50:

            score += 35

            reason += (
                " Moderate rainfall is favorable."
            )

        elif rain_probability > 70:

            score += 10

            reason += (
                " Heavy rainfall may reduce suitability."
            )

        else:

            score += 20

            reason += (
                " Current rainfall is moderately suitable."
            )


    elif crop == "Banana":

        if 25 <= temperature <= 35:

            score += 40

            reason = (
                "Warm temperature is suitable for banana."
            )


        if rain_probability >= 40:

            score += 35

            reason += (
                " Adequate rainfall supports banana growth."
            )

        else:

            score += 15

            reason += (
                " Additional irrigation may be required."
            )


    elif crop == "Watermelon":

        if 25 <= temperature <= 35:

            score += 40

            reason = (
                "Warm weather is favorable for watermelon."
            )


        if rain_probability <= 50:

            score += 35

            reason += (
                " Lower rainfall helps avoid excess moisture."
            )

        else:

            score += 10

            reason += (
                " Higher rainfall may reduce suitability."
            )


    return score, reason


# =========================================================
# SMART CROP PLANNER
# =========================================================

@app.route(
    "/planner",
    methods=["GET", "POST"]
)
def planner():

    crops = []

    weather = None

    weather_available = False

    temperature = 0

    rain_probability = 0

    best_match = None

    best_match_reason = ""

    irrigation_warning = ""

    crop_results = []


    if request.method == "POST":

        soil = request.form.get(
            "soil"
        )

        water = request.form.get(
            "water"
        )

        region = request.form.get(
            "region"
        )

        budget = request.form.get(
            "budget"
        )


        weather_data = get_weather(
            region
        )


        weather = weather_data["current"]

        daily = weather_data["daily"]

        weather_available = weather_data["available"]


        temperature = weather[
            "temperature_2m"
        ]


        rain_probability = (
            daily[
                "precipitation_probability_max"
            ][0]
        )


        if rain_probability >= 70:

            irrigation_warning = (
                "🌧️ கனமழைக்கான வாய்ப்பு உள்ளது. "
                "தேவையற்ற நீர்ப்பாசனத்தை தவிர்த்து, "
                "வயலில் நீர் வடிகால் வசதியை சரிபார்க்கவும்."
            )

        elif rain_probability >= 50:

            irrigation_warning = (
                "🌦️ மிதமான மழைக்கான வாய்ப்பு உள்ளது. "
                "நீர்ப்பாசனம் செய்வதற்கு முன் "
                "மண்ணின் ஈரப்பதத்தை சரிபார்க்கவும்."
            )

        else:

            irrigation_warning = (
                "💧 மழை வாய்ப்பு குறைவாக உள்ளது. "
                "மண்ணின் ஈரப்பதத்தைப் பொறுத்து "
                "நீர்ப்பாசனம் தேவைப்படலாம்."
            )


        # -------------------------------------------------
        # SOIL + WATER CROP RECOMMENDATION
        # -------------------------------------------------

        if soil == "red":

            if water in ["well", "borewell"]:

                crops.extend([
                    "Groundnut",
                    "Millet",
                    "Cotton"
                ])

            elif water == "rain":

                crops.extend([
                    "Groundnut",
                    "Millet"
                ])

            else:

                crops.extend([
                    "Groundnut",
                    "Millet"
                ])


        elif soil == "black":

            if water in [
                "well",
                "borewell",
                "canal",
                "river"
            ]:

                crops.extend([
                    "Cotton",
                    "Black Gram",
                    "Sugarcane"
                ])

            else:

                crops.extend([
                    "Cotton",
                    "Black Gram"
                ])


        elif soil == "alluvial":

            if water in [
                "canal",
                "river",
                "well",
                "borewell"
            ]:

                crops.extend([
                    "Rice",
                    "Sugarcane",
                    "Banana"
                ])

            elif water == "rain":

                crops.extend([
                    "Rice",
                    "Black Gram"
                ])

            else:

                crops.extend([
                    "Rice",
                    "Black Gram"
                ])


        elif soil == "sandy":

            if water in [
                "well",
                "borewell"
            ]:

                crops.extend([
                    "Groundnut",
                    "Watermelon",
                    "Millet"
                ])

            else:

                crops.extend([
                    "Groundnut",
                    "Millet"
                ])


        elif soil == "clay":

            if water in [
                "canal",
                "river",
                "well",
                "borewell"
            ]:

                crops.extend([
                    "Rice",
                    "Sugarcane"
                ])

            else:

                crops.extend([
                    "Rice"
                ])


        # -------------------------------------------------
        # REGION PRIORITY
        # -------------------------------------------------

        if region == "cuddalore":

            preferred_crops = [
                "Rice",
                "Groundnut",
                "Black Gram",
                "Sugarcane"
            ]

        elif region == "villupuram":

            preferred_crops = [
                "Groundnut",
                "Rice",
                "Black Gram",
                "Millet"
            ]

        elif region == "chennai":

            preferred_crops = [
                "Millet",
                "Groundnut",
                "Black Gram"
            ]

        elif region == "pondicherry":

            preferred_crops = [
                "Rice",
                "Groundnut",
                "Black Gram",
                "Sugarcane"
            ]

        else:

            preferred_crops = [
                "Rice",
                "Groundnut",
                "Millet",
                "Black Gram"
            ]


        crops = sorted(
            set(crops),
            key=lambda crop:
                (
                    preferred_crops.index(crop)
                    if crop in preferred_crops
                    else 999
                )
        )


        # -------------------------------------------------
        # BUDGET
        # -------------------------------------------------

        if budget:

            try:

                budget = float(
                    budget
                )

            except (
                ValueError,
                TypeError
            ):

                budget = 0

        else:

            budget = 0


        if budget > 0:

            if budget < 10000:

                low_budget_crops = [
                    "Millet",
                    "Black Gram",
                    "Groundnut"
                ]

                filtered_crops = [
                    crop
                    for crop in crops
                    if crop in low_budget_crops
                ]

                if filtered_crops:

                    crops = filtered_crops


            elif 10000 <= budget < 30000:

                medium_budget_crops = [
                    "Groundnut",
                    "Black Gram",
                    "Millet",
                    "Rice",
                    "Cotton"
                ]

                filtered_crops = [
                    crop
                    for crop in crops
                    if crop in medium_budget_crops
                ]

                if filtered_crops:

                    crops = filtered_crops


        if not crops:

            crops = [
                "Groundnut",
                "Black Gram",
                "Millet"
            ]


        # -------------------------------------------------
        # WEATHER RANKING
        # -------------------------------------------------

        scored_crops = []


        for crop in crops:

            score, reason = get_weather_score(
                crop,
                temperature,
                rain_probability
            )

            scored_crops.append({

                "name": crop,

                "score": score,

                "reason": reason
            })


        scored_crops.sort(
            key=lambda item: item["score"],
            reverse=True
        )


        crop_results = scored_crops


        if crop_results:

            best_match = (
                crop_results[0]["name"]
            )

            best_match_reason = (
                crop_results[0]["reason"]
                +
                " This crop currently has "
                "the highest weather suitability score."
            )


        print(
            "--------------------------------"
        )

        print(
            "SMART CROP PLANNER"
        )

        print(
            "Soil:",
            soil
        )

        print(
            "Water:",
            water
        )

        print(
            "Region:",
            region
        )

        print(
            "Budget:",
            budget
        )

        print(
            "Temperature:",
            temperature
        )

        print(
            "Rain Probability:",
            rain_probability
        )

        print(
            "Crops:",
            crops
        )

        print(
            "Best Match:",
            best_match
        )

        print(
            "--------------------------------"
        )


    return render_template(

        "planner.html",

        crops=crops,

        weather=weather,

        weather_available=weather_available,

        temperature=temperature,

        rain_probability=rain_probability,

        irrigation_warning=irrigation_warning,

        crop_results=crop_results,

        best_match=best_match,

        best_match_reason=best_match_reason
    )


# =========================================================
# TAMIL ASSISTANT
# =========================================================

def tamil_assistant_response(question):

    q = question.strip().lower()


    if any(word in q for word in [
        "வணக்கம்",
        "hello",
        "hi",
        "ஹலோ"
    ]):

        return (
            "வணக்கம்! 🌱\n\n"
            "நான் விதைமதி தமிழ் விவசாய உதவியாளர்.\n"
            "வானிலை, நீர்ப்பாசனம், பயிர்கள் மற்றும் "
            "பயிர் நோய்கள் பற்றிய தகவல்களை கேட்கலாம்."
        )


    if any(word in q for word in [
        "நோய்",
        "நோயை",
        "நோய்கள்",
        "நோய் கண்டறி",
        "நோய் கண்டறிவது",
        "disease",
        "disease detection",
        "plant disease",
        "crop disease",
        "இலை நோய்"
    ]):

        return (
            "🌿 பயிர் நோயை கண்டறிய, விதைமதியின் "
            "“பயிர் நோய் கண்டறிதல்” பகுதியில் தெளிவான "
            "இலை புகைப்படத்தை பதிவேற்றுங்கள்.\n\n"
            "🤖 AI படம் பகுப்பாய்வு செய்து சாத்தியமான "
            "நோயை மற்றும் confidence-ஐ காட்டும்.\n\n"
            "⚠️ AI கணிப்பு ஒரு உதவி தகவல் மட்டுமே. "
            "நோய் உறுதிப்படுத்த தேவையானால் வேளாண்மை "
            "நிபுணரிடம் ஆலோசனை பெறுங்கள்."
        )


    if any(word in q for word in [
        "வானிலை",
        "weather",
        "மழை வரும்",
        "மழை",
        "temperature",
        "வெப்பநிலை",
        "இன்று வானிலை"
    ]):

        try:

            weather_data = get_weather(
                "cuddalore"
            )

            current = weather_data.get(
                "current",
                {}
            )

            daily = weather_data.get(
                "daily",
                {}
            )

            temperature = current.get(
                "temperature_2m",
                "தகவல் இல்லை"
            )

            humidity = current.get(
                "relative_humidity_2m",
                "தகவல் இல்லை"
            )

            rain_probability_list = daily.get(
                "precipitation_probability_max",
                []
            )

            rain_probability = (
                rain_probability_list[0]
                if rain_probability_list
                else "தகவல் இல்லை"
            )

            rainfall = current.get(
                "precipitation",
                0
            )

            wind = current.get(
                "wind_speed_10m",
                "தகவல் இல்லை"
            )

            return (
                "🌦️ இன்றைய வானிலை:\n\n"
                f"🌡️ வெப்பநிலை: {temperature}°C\n"
                f"💧 ஈரப்பதம்: {humidity}%\n"
                f"🌧️ மழை அளவு: {rainfall} mm\n"
                f"☔ மழை வாய்ப்பு: {rain_probability}%\n"
                f"💨 காற்றின் வேகம்: {wind} km/h"
            )

        except Exception as e:

            print(
                "Assistant weather error:",
                e
            )

            return (
                "⚠️ தற்போது நேரடி வானிலை தகவலை "
                "பெற முடியவில்லை.\n\n"
                "சிறிது நேரம் கழித்து மீண்டும் முயற்சிக்கவும்."
            )


    if any(word in q for word in [
        "நீர்ப்பாசனம்",
        "நீர் பாய்ச்ச",
        "தண்ணீர் பாய்ச்ச",
        "தண்ணீர்",
        "irrigation",
        "water the crop",
        "water crop"
    ]):

        try:

            weather_data = get_weather(
                "cuddalore"
            )

            current = weather_data.get(
                "current",
                {}
            )

            daily = weather_data.get(
                "daily",
                {}
            )

            temperature = current.get(
                "temperature_2m",
                0
            )

            rain_probability_list = daily.get(
                "precipitation_probability_max",
                []
            )

            rain_probability = (
                rain_probability_list[0]
                if rain_probability_list
                else 0
            )


            if rain_probability >= 60:

                return (
                    "☔ இன்று மழை வாய்ப்பு அதிகமாக உள்ளது "
                    f"({rain_probability}%).\n\n"
                    "💧 உடனடியாக அதிகமாக நீர்ப்பாசனம் "
                    "செய்வதை தவிர்ப்பது நல்லது.\n"
                    "மண்ணின் ஈரப்பதத்தை பார்த்து முடிவு செய்யுங்கள்."
                )


            if (
                temperature >= 30
                and
                rain_probability < 40
            ):

                return (
                    "💧 இன்று நீர்ப்பாசனம் தேவையாக இருக்கலாம்.\n\n"
                    f"🌡️ வெப்பநிலை: {temperature}°C\n"
                    f"☔ மழை வாய்ப்பு: {rain_probability}%\n\n"
                    "மண்ணின் ஈரப்பதத்தையும் பயிரின் நிலையையும் "
                    "பார்த்து தேவையான அளவு மட்டும் நீர் வழங்குங்கள்."
                )


            return (
                "💧 இன்று நீர்ப்பாசனம் செய்வதற்கு முன் "
                "மண்ணின் ஈரப்பதத்தை சரிபார்க்கவும்.\n\n"
                f"🌡️ வெப்பநிலை: {temperature}°C\n"
                f"☔ மழை வாய்ப்பு: {rain_probability}%\n\n"
                "மண் ஏற்கனவே ஈரமாக இருந்தால் கூடுதல் "
                "நீர்ப்பாசனம் தேவையில்லை."
            )


        except Exception as e:

            print(
                "Assistant irrigation error:",
                e
            )

            return (
                "⚠️ நீர்ப்பாசனத்திற்கான நேரடி வானிலை தகவலை "
                "பெற முடியவில்லை.\n\n"
                "மண்ணின் ஈரப்பதத்தை பார்த்து நீர்ப்பாசனம் செய்யவும்."
            )


    if (
        "சிவப்பு மண்" in q
        or
        "red soil" in q
        or
        "red soil crop" in q
    ):

        return (
            "🌱 சிவப்பு மண்ணில் பொதுவாக நல்ல வடிகால் "
            "தேவைப்படும் பயிர்கள் ஏற்றதாக இருக்கலாம்.\n\n"
            "🌾 நிலக்கடலை\n"
            "🌾 கம்பு / சிறுதானியங்கள்\n"
            "🌾 பருப்பு வகைகள்\n\n"
            "ஆனால் சரியான பயிரை தேர்வு செய்ய மண் தன்மை, "
            "நீர்வசதி, பருவம் மற்றும் உள்ளூர் வானிலை "
            "ஆகியவற்றையும் கருத்தில் கொள்ள வேண்டும்."
        )


    if any(word in q for word in [
        "பயிர்",
        "பயிர்கள்",
        "crop",
        "crops",
        "விதை",
        "விவசாயம்",
        "farming"
    ]):

        return (
            "🌱 சரியான பயிரை தேர்வு செய்ய மண் வகை, "
            "நீர்வசதி, பகுதி, பருவம் மற்றும் வானிலை "
            "ஆகியவற்றை கருத்தில் கொள்ள வேண்டும்.\n\n"
            "விதைமதி Smart Crop Planner-ல் இந்த "
            "தகவல்களை உள்ளிட்டு பொருத்தமான பயிர்களை "
            "பார்க்கலாம்."
        )


    if any(word in q for word in [
        "எப்படி",
        "உதவி",
        "help",
        "farming tips",
        "விவசாய குறிப்புகள்",
        "விவசாய குறிப்பு"
    ]):

        return (
            "🌾 நான் உங்களுக்கு வானிலை, நீர்ப்பாசனம், "
            "பயிர் தேர்வு மற்றும் பயிர் நோய் கண்டறிதல் "
            "பற்றி உதவ முடியும்.\n\n"
            "உதாரணமாக:\n"
            "• இன்று வானிலை எப்படி?\n"
            "• இன்று நீர்ப்பாசனம் செய்யலாமா?\n"
            "• சிவப்பு மண்ணுக்கு என்ன பயிர் ஏற்றது?\n"
            "• பயிர் நோயை எப்படி கண்டறிவது?"
        )


    return (
        "🌱 உங்கள் கேள்வியை இன்னும் கொஞ்சம் தெளிவாக "
        "கேட்கவும்.\n\n"
        "வானிலை, நீர்ப்பாசனம், பயிர் தேர்வு அல்லது "
        "பயிர் நோய் பற்றி கேட்கலாம்."
    )


# =========================================================
# TAMIL ASSISTANT PAGE
# =========================================================

@app.route(
    "/assistant",
    methods=["GET", "POST"]
)
def assistant():

    question = ""

    answer = ""


    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        ).strip()


        if question:

            answer = tamil_assistant_response(
                question
            )


    return render_template(
        "assistant.html",
        question=question,
        answer=answer
    )


# =========================================================
# TAMIL VOICE OUTPUT
# =========================================================

@app.route(
    "/assistant/speak",
    methods=["POST"]
)
def assistant_speak():

    try:

        text = request.form.get(
            "text",
            ""
        ).strip()


        if not text:

            data = request.get_json(
                silent=True
            ) or {}

            text = str(
                data.get(
                    "text",
                    ""
                )
            ).strip()


        if not text:

            return (
                "No text provided",
                400
            )


        audio_folder = os.path.join(
            "static",
            "audio"
        )


        os.makedirs(
            audio_folder,
            exist_ok=True
        )


        audio_file = os.path.join(

            audio_folder,

            f"tamil_voice_{uuid.uuid4().hex}.mp3"
        )


        async def generate_voice():

            communicate = edge_tts.Communicate(
                text,
                "ta-IN-PallaviNeural"
            )

            await communicate.save(
                audio_file
            )


        asyncio.run(
            generate_voice()
        )


        return send_file(
            audio_file,
            mimetype="audio/mpeg"
        )


    except Exception as e:

        print(
            "Tamil voice error:",
            e
        )

        return (
            "Tamil voice generation failed",
            500
        )


# =========================================================
# OFFLINE AGRICULTURE TIPS
# =========================================================

@app.route("/offline-tips")
def offline_tips():

    language = request.args.get(
        "lang",
        "ta"
    )


    if language == "en":

        tips = OFFLINE_TIPS_ENGLISH

    else:

        tips = OFFLINE_TIPS


    return render_template(

        "offline_tips.html",

        offline_tips=tips,

        language=language
    )


# =========================================================
# RUN FLASK APP
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )