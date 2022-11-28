from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
import os

from images.models import pI, isP
from learningModel.isPModel import isPWoker
from learningModel.pGenerationModel import pGenerationWoker

from myP.settings import STATICFILES_DIRS

ispworker = isPWoker()
pgenerationwoker = pGenerationWoker()


def index(request):
    template = loader.get_template('index.html')
    pITableLength = pI.objects.all().count()
    isPTableLength = isP.objects.all().count()
    context = {
        'pITableLength': pITableLength,
        'isPTableLength': isPTableLength
    }
    return HttpResponse(template.render(context, request))


def updateDB(request):
    pI.objects.all().delete()

    def getListOfFiles(dirName):
        # create a list of file and sub directories
        # names in the given directory
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(fullPath):
                allFiles = allFiles + getListOfFiles(fullPath)
            elif fullPath.endswith('.png') or fullPath.endswith('.jpg') or fullPath.endswith('.jpeg'):
                allFiles.append(os.path.relpath(fullPath, STATICFILES_DIRS[1]))

        return allFiles

    for p in getListOfFiles(STATICFILES_DIRS[1]):
        pathObj = pI(path=p)
        pathObj.save()

    return HttpResponseRedirect('/images/')


def show(request, withPrediciton):
    imageToShow = pI.objects.values_list('path').order_by('?').first()[0]
    isPPrediction = ispworker.predict(imageToShow)
    if withPrediciton == 1:
        count = 0
        while isPPrediction < 0.5 and count < 10:
            imageToShow = pI.objects.values_list(
                'path').order_by('?').first()[0]
            isPPrediction = ispworker.predict(imageToShow)
            count += 1
    template = loader.get_template('show.html')
    context = {
        'imageToShow': imageToShow,
        'isPPrediction': isPPrediction,
        'withPrediciton': withPrediciton
    }
    return HttpResponse(template.render(context, request))


def labelImage(request, withPrediciton):
    imagePath, imageLabel = request.POST.get('path'), request.POST.get('isP')
    isPObj = isP(path=imagePath, label=imageLabel)
    isPObj.save()
    return HttpResponseRedirect('/images/show/'+str(withPrediciton))


def learnImage(request):
    ispworker.train()
    pgenerationwoker.train()
    return HttpResponseRedirect(reverse('index'))


def generateImage(request):
    pgenerationwoker.generateImg()
    template = loader.get_template('generate.html')
    context = {
        'imageToShow': "generated0.jpg"
    }
    return HttpResponse(template.render(context, request))


def clearDataset(request):
    isP.objects.all().delete()
    return HttpResponseRedirect(reverse('index'))
