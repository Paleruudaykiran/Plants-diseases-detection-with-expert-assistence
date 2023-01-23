from django.shortcuts import render
from django.core.files.storage import FileSystemStorage 
from django.conf import settings
import os
import cv2 
import numpy as np
import pickle
from keras.models import load_model 
rice = [
        {
         'Symptoms' : 'The symptoms appear on leaf blades and sheaths as small linear, water-soaked areas that soon elongated and coalesce into irregular, narrow, yellowish, or brownish stripes. Severe infection cause leaves to turn yellow and die from the tip to downward. They also retard spike elongated and cause blushing. The small lesion from on the kernels as well.',
         'Control Measure' : 'The main control measures are the use of disease-free or treated seeds and crop rotation. Stubble and straw should be burned after paddy harvesting on unhealthy soil. After seeing the disease, an additional 20 kg of potash fertilizer per acre should be applied to the water. Spraying chelated zinc at the rate of 1 gram per liter of water reduces the severity of the disease.',
         'Recurrence of the disease' : 'The disease develops mainly in rainy, damp weather. The bacteria overwinter on the seed and crop residues and are spread by rain, direct contact with an insect.',
        },
        {
         'Symptoms' : 'All parts of the plants are being infected except roots. Sometimes the necrotic   lesions are found to the present even on the first emerging leaves. In severe cause, the seedling   are blighted and become dead. The most characteristics symptoms of the disease is the development of dark brown spot on the upper surface of the leaf lamina. The spots are developed on the leaf blade and leaf sheath. The spots are various shapes and sizes. They are isolated dark-brown oval and scattered through out the leaf blade and leaf sheath. The spots developed also on leaf sheath and culm below the ear. The base of the panicle becomes shrunk and discolured. The inflorescence and individual spikelets also become infected. If the inflorescence is attacked at an   early stage , it fails to develop. The grains forms inside the affectd spikelets become discoloured',
         'Control Measure' : 'Seeds are treated with agrosan GN, organo mercurials. In 50-52ᵒc temperature for 10 minutes hot water treatment also done in severe cases.Spraying of bordex mixture, perenox, Dihane-78 has been found to very effective.',
         'Nature & recurrence of Brown spot of rice' : 'Disease is primarily seed borned. The fungus is found as dormant mycelium in the husk and on the kernel and often it infects the germinating grain. Secondary  infection is caused by air borne conidia. Generally, such conidia are produced either from primary infected plants or from other sources.',
        },
        {
        'Symptoms' : 'This is a minor fungal disease in which small slightly raised black spots develop primarily on the leaves. Raised spots or pustules break open releasing air-borne spores. Infection is often heavy enough to kill tips of leaves. Leaf smut occurs late in the growing season and causes little or no economic loss.',
        'Rice Leaf Smut Information' : 'What causes rice leaf smut is a fungus called Entyloma oryzae. Fortunately for your garden, if you see its signs, this infection is usually minor. It is widespread where rice is grown, but leaf smut does not often cause serious damage. However, leaf smut can make your rice vulnerable to other diseases, and ultimately this can cause a yield reduction. The characteristic sign of rice with leaf smut is the presence of small black spots on the leaves. They are slightly raised and angular and give the leaves the appearance of having been sprinkled with ground pepper. Coverage by these spots is most complete on the oldest leaves. The tips of some leaves with the most infection may die.',
        'Prevention' : 'In most situations, there is no major loss caused by rice leaf smut, so treatment is not usually given. However, it can be a good idea to use good general management practices to prevent the infection or keep it in check, and to keep plants healthy overall.As with many other fungal infections, this one is spread by infected plant material in the soil. When healthy leaves contact the water or ground with old diseased leaves, they can become infected. Cleaning up debris at the end of each growing season can prevent spread of leaf smut. Keeping a good nutrient balance is also important, as high nitrogen levels increases the incidence of the disease.',
        },
]
potato = [ 
    {
        'Symptoms' : 'At first, we see small, pale brown, and scattered spots on the leaflets. There is a deep greenish-blue growth of fungus on these spots. First attack appears on the leaves near the soil surface, now it progresses upward.In the necrotic area of the spots, some concentric ridges develop which form a target board. This is the most characteristic symptom of ealry blight of potato.',
        'Management' : 'Crop rotation and field sanitation are essential for effective control of the disease.Destroy dead haulm by burning it.Maintain good plant vigour.',
        'Causes' : 'It is due to interruption of fugal growth. Unfavorable weather conditions are the cause of interruption of fungal growth. Presence of a narrow chlorotic zone around the spots is also a symptom of this disease. Further, this chlorotic zone turns into the normal green. Size of this chlorotic zone depends on the size of target board.',
    },
    {
        'Precautions' : 'Protect plants from early blight, late blight and  Canker Stem Disease',
    },
    {
        'Symptoms' : 'Affected leaves of the plant wilt and turn brown within weeks. Sometimes, spots appear on the underside of leaves.The stems become black from the tips and dry out eventually.Infected tubers have dry brown-colored spots on their skins and flesh.',
        'Prevention' : 'Destroy or bury all crop debris and tubers at the end of each season.Use disease-free varieties of seed potatoes to grow the plants.Use a thick layer of mulch to prevent tuber infection',
        'Pathogen' : 'The late blight disease is caused by a fungus-like organism Phytophthora infestans. The pathogen survives in plant debris in the soil and spreads mainly through the soil and infected seed tubers.The pathogen becomes highly active in humid conditions, low temperature and leaf wetness.',
   }
]


def predict_rice_disease(root_dir,file_name) :
    class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
    file_path = os.path.join(root_dir,file_name)
    img_height,img_width=256,256
    image = cv2.imread(str(file_path)) 
    image_resized = cv2.resize(image,(img_height,img_width)) 
    image=np.expand_dims(image_resized,axis=0) 
    print(image)
    model = load_model('F:/FYP-code/static/dl_models/predict_rice.h5')
    pred = model.predict(image) 
    output_class = class_names[np.argmax(pred)] 
    accuracy = pred[0][np.argmax(pred)]
    
    return output_class,accuracy
def prediction_rice(request) :
    context = {}
    if request.method == 'POST' : 
       upload_file = request.FILES['file'] 
       fs = FileSystemStorage() 

       mypath = settings.MEDIA_ROOT
       for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))
    
       name = fs.save(upload_file.name,upload_file)
       pred,acc = predict_rice_disease(mypath,name) 
       context['url'] = fs.url(name)  
       context['disease'] = pred
       
       if len(str(acc)) >= 4 :
        context['accuracy'] = str(acc*100)[:4] 
       else :
        context['accuracy'] = acc*100 
       
       indices_diseases = {
        'Bacterial leaf blight' : 0,
        'Brown spot' : 1,
        'Leaf smut' : 2,
       }
       context['info'] = rice[indices_diseases[pred]] 
    return render(request,'price.html',context)


def predict_potato_disease(root_dir,file_name) :
    class_names = ['Early Blight', 'Healthy', 'Late Blight']
    file_path = os.path.join(root_dir,file_name)
    img_height,img_width=256,256
    image = cv2.imread(str(file_path)) 
    image_resized = cv2.resize(image,(img_height,img_width)) 
    image=np.expand_dims(image_resized,axis=0) 
    model = load_model('F:/FYP-code/static/dl_models/predict_potato.h5')
    pred = model.predict(image) 
    output_class = class_names[np.argmax(pred)] 
    accuracy = pred[0][np.argmax(pred)]
    return output_class,accuracy
def prediction_potato(request) :
    context = {}
    if request.method == 'POST' : 
       upload_file = request.FILES['file'] 
       fs = FileSystemStorage() 

       mypath = settings.MEDIA_ROOT
       for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))
    
       name = fs.save(upload_file.name,upload_file)
       pred,acc = predict_potato_disease(mypath,name) 
       context['url'] = fs.url(name)  
       context['disease'] = pred
       
       if len(str(acc)) >= 4 :
        context['accuracy'] = str(acc*100)[:4] 
       else :
        context['accuracy'] = acc*100 
       
       indices_diseases = {
        'Early Blight' : 0,
        'Healthy' : 1,
        'Late Blight' : 2
       }
       context['info'] = potato[indices_diseases[pred]] 
    return render(request,'price.html',context)


def predict_tomato_disease(root_dir,file_name) :
    class_names = ['"Bacterial_spot"', 'Early_blight', 'Late_blight','Leaf_Mold',"Septoria_leaf_spot","Spider_mites Two-spotted_spider_mite","Target_Spot","Tomato_Yellow_Leaf_Curl_Virus","Tomato_mosaic_virus","Healthy"]
    file_path = os.path.join(root_dir,file_name)
    img_height,img_width=224,224
    image = cv2.imread(str(file_path)) 
    image_resized = cv2.resize(image,(img_height,img_width)) 
    image_resized = image_resized/255 
    image=np.expand_dims(image_resized,axis=0) 
    model = load_model('F:/FYP-code/static/dl_models/predict_tomato.h5')
    pred = model.predict(image) 
    output_class = class_names[np.argmax(pred)] 
    accuracy = pred[0][np.argmax(pred)]
    return output_class,accuracy
def prediction_tomato(request) :
    context = {}
    if request.method == 'POST' : 
       upload_file = request.FILES['file'] 
       fs = FileSystemStorage() 

       mypath = settings.MEDIA_ROOT
       for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))
    
       name = fs.save(upload_file.name,upload_file)
       pred,acc = predict_tomato_disease(mypath,name) 
       context['url'] = fs.url(name)  
       context['disease'] = pred
       
       if len(str(acc)) >= 4 :
        context['accuracy'] = str(acc*100)[:4] 
       else :
        context['accuracy'] = acc*100 
       
       indices_diseases = {
        "Bacterial_spot" : 0,
        "Early_blight" : 1,
        "Late_blight" : 2,
        "Leaf_Mold" : 3,
        "Septoria_leaf_spot" : 4,
        "Spider_mites Two-spotted_spider_mite" : 5,
        "Target_Spot" : 6,
        "Tomato_Yellow_Leaf_Curl_Virus" : 7,
        "Tomato_mosaic_virus" : 8,
        "Healthy" : 9,
       }
       context['info'] = potato[0] 
    return render(request,'price.html',context)

def predict_cotton_disease(root_dir,file_name) :
    class_names = ['Diseased Leaf','Diseased plant','Fresh Leaf',"Fresh platn",]
    file_path = os.path.join(root_dir,file_name)
    img_height,img_width=224,224
    image = cv2.imread(str(file_path)) 
    image_resized = cv2.resize(image,(img_height,img_width)) 
    image_resized = image_resized/255 
    image=np.expand_dims(image_resized,axis=0) 
    model = load_model('F:/FYP-code/static/dl_models/predict_cotton.h5')
    pred = model.predict(image) 
    output_class = class_names[np.argmax(pred)] 
    accuracy = pred[0][np.argmax(pred)]
    return output_class,accuracy
def prediction_cotton(request) :
    context = {}
    if request.method == 'POST' : 
       upload_file = request.FILES['file'] 
       fs = FileSystemStorage() 

       mypath = settings.MEDIA_ROOT
       for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))
    
       name = fs.save(upload_file.name,upload_file)
       pred,acc = predict_cotton_disease(mypath,name) 
       context['url'] = fs.url(name)  
       context['disease'] = pred
       
       if len(str(acc)) >= 4 :
        context['accuracy'] = str(acc*100)[:4] 
       else :
        context['accuracy'] = acc*100 
       
       indices_diseases = {
        "Diseased Leaf" : 0,
        "Diseased Plant" : 1,
        "Fresh Leaf" : 2,
        "Fresh Plant" : 3,
       }
       context['info'] = potato[0] 
    return render(request,'price.html',context)


def home(request) : 
    return render(request,'home.html')