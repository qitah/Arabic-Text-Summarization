from django.shortcuts import render
import arabic_nlp.arabic_script.elements as ase
from tashaphyne.stemming import ArabicLightStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import scipy as sp 
import networkx as nx
from collections import OrderedDict
import qalsadi.lemmatizer 
from typing import Coroutine

def index(request):
    return render(request,"index.html")

def Text(request):
    if request.method == 'POST':
        upload = request.FILES['text'].read()
        original_data = upload.decode('utf-8')
        Edited_data = original_data

        Original_paragraph = original_data.split(".")
        countt = 0

        for i in Original_paragraph:
            Original_paragraph[countt] = i.strip()
            countt = countt + 1
        
        Original_paragraph = Original_paragraph[0:-1]

        def encoding_cleanup():
            raise NotImplementedError


        def tatweel_removal(text):
        
            if text is None:
                return None

            return text.replace(ase.TATWEEL, '')


        def diacritic_removal(text):
           
            if text is None:
                return None

            for diacritic in ase.DIACRITICS:
                text = text.replace(diacritic, '')

            return text


        def punctuation_removal(text):
          
            if text is None:
                return None

            for mark in ase.PUNCTUATION_MARKS:
                if mark in ase.NUMBERS_PUNCTUATION_MARKS:
                    continue

                text = text.replace(mark, '')

            return text


        def letter_normalization(text, egyptian=False):
           
            if text is None:
                return None

            if egyptian:
                text = text.replace(ase.ALIF_MAQSURA, 'ي')

            for form in ase.ALEF_HAMZA_FORMS:
                text = text.replace(form, 'ا')

            text = text.replace(ase.TA_MARBUTA, 'ه')

            for form in ase.NON_ALIF_HAMZA_FORMS:
                text = text.replace(form, ase.HAMZA)

            return text


        def clean_text(text):
                
                if text is None:
                    return None

                # Remove whitespace characters from the beginning and the end
                text = text.strip()

                for letter in text:
                    if letter not in ase.LETTERS and letter != ' ':
                        text = text.replace(letter, '')

                return text
        
        Edited_data = diacritic_removal(Edited_data)
        Edited_data = tatweel_removal(Edited_data)
        Edited_data = punctuation_removal(Edited_data)
        Edited_data = letter_normalization(Edited_data, egyptian=False)
        Edited_data = clean_text(Edited_data)

        Edited_paragraph = Edited_data.split(".")

        count = 0
        for i in Edited_paragraph:
            Edited_paragraph[count] = i.strip()
            count = count + 1
        
        sss = ['', '،','ء','ءَ','آ','آب','آذار','آض','آل','آمينَ','آناء','آنفا','آه','آهاً','آهٍ','آهِ','أ','أبدا','أبريل','أبو','أبٌ','أجل','أجمع','أحد','أخبر',
        'أخذ','أخو','أخٌ','أربع','أربعاء','أربعة','أربعمئة','أربعمائة','أرى','أسكن','أصبح','أصلا','أضحى','أطعم','أعطى','أعلم','أغسطس','أفريل','أفعل','به','أفٍّ','أقبل','أكتوبر','أل','ألا','ألف','ألفى','أم','أما','أمام','أمامك','أمامكَ',
        'أمد','أمس','أمسى','أمّا','أن','أنا','أنبأ','أنت','أنتم','أنتما','أنتن','أنتِ','أنشأ','أنه','أنًّ','أنّى','أهلا','أو','أوت','أوشك','أول','أولئك','أولاء','أولالك','أوّهْ','أى','أي','أيا','أيار','أيضا',
        'أيلول','أين','أيّ','أيّان','أُفٍّ','ؤ','إحدى','إذ','إذا','إذاً','إذما','إذن','إزاء','إلى','إلي','إليكم','إليكما','إليكنّ','إليكَ','إلَيْكَ','إلّا','إمّا','إن','إنَّ','إى','إياك','إياكم','إياكما','إياكن','إيانا','إياه',
        'إياها','إياهم','إياهما','إياهن','إياي','إيهٍ','ئ','ا','ا?','ا?ى','االا','االتى','ابتدأ','ابين','اتخذ','اثر','اثنا','اثنان','اثني','اثنين','اجل','احد','اخرى','اخلولق','اذا','اربعة','اربعون','اربعين','ارتدّ','استحال','اصبح',
        'اضحى','اطار','اعادة','اعلنت','اف','اكثر','اكد','الآن','الألاء','الألى','الا','الاخيرة','الان','الاول','الاولى','التى','التي','الثاني','الثانية','الحالي','الذاتي','الذى','الذي','الذين','السابق','الف','اللاتي','اللتان','اللتيا','اللتين','اللذان','اللذين',
        'اللواتي','الماضي','المقبل','الوقت','الى','الي','اليه','اليها','اليوم','اما','امام','امس','امسى','ان','انبرى','انقلب','انه','انها','او','اول','اي','ايار','ايام','ايضا','ب','بؤسا','بإن','بئس','باء','بات','باسم',
        'بان','بخٍ','بد','بدلا','برس','بسبب','بسّ','بشكل','بضع','بطآن','بعد','بعدا','بعض','بغتة','بل','بلى','بن','به','بها','بهذا','بيد','بين','بَسْ','بَلْهَ','ة','ت','تاء','تارة','تاسع','تانِ',
        'تانِك','تبدّل','تجاه','تحت','تحوّل','تخذ','ترك','تسع','تسعة','تسعمئة','تسعمائة','تسعون','تسعين','تشرين','تعسا','تعلَّم','تفعلان','تفعلون','تفعلين','تكون','تلقاء','تلك','تم','تموز','تينك','تَيْنِ','تِه','تِي','ث','ثاء','ثالث','ثامن','ثان','ثاني','ثلاث','ثلاثاء','ثلاثة','ثلاثمئة','ثلاثمائة','ثلاثون','ثلاثين','ثم','ثمان','ثمانمئة','ثمانون','ثماني','ثمانية','ثمانين','ثمنمئة','ثمَّ','ثمّ','ثمّة','ج','جانفي','جدا','جعل','جلل','جمعة','جميع','جنيه','جوان','جويلية','جير','جيم','ح','حاء','حادي','حار','حاشا','حاليا','حاي','حبذا','حبيب','حتى','حجا','حدَث','حرى','حزيران','حسب','حقا','حمدا','حمو','حمٌ','حوالى','حول','حيث','حيثما','حين','حيَّ','حَذارِ','خ','خاء','خاصة','خال','خامس','خبَّر','خلا','خلافا','خلال','خلف','خمس','خمسة','خمسمئة',
        'خمسمائة','خمسون','خمسين','خميس','د','دال','درهم','درى','دواليك','دولار','دون','دونك','ديسمبر','دينار','ذ','ذا','ذات','ذاك','ذال','ذانك','ذانِ','ذلك','ذهب','ذو','ذيت','ذينك','ذَيْنِ','ذِه','ذِي','ر','رأى','راء','رابع','راح','رجع','رزق','رويدك','ريال','ريث','رُبَّ','ز','زاي','زعم','زود','زيارة','س','ساء','سابع','سادس','سبت','سبتمبر','سبحان','سبع','سبعة','سبعمئة','سبعمائة','سبعون','سبعين','ست','ستة','ستكون','ستمئة','ستمائة','ستون','ستين','سحقا','سرا','سرعان','سقى','سمعا','سنة','سنتيم','سنوات','سوف','سوى','سين','ش','شباط','شبه','شتانَ','شخصا','شرع','شمال','شيكل','شين','شَتَّانَ','ص','صاد','صار','صباح','صبر','صبرا',
        'صدقا','صراحة','صفر','صهٍ','صهْ','ض','ضاد','ضحوة','ضد','ضمن','ط','طاء','طاق','طالما','طرا','طفق','طَق','ظ','ظاء','ظل','ظلّ','ظنَّ','ع','عاد','عاشر','عام','عاما','عامة','عجبا','عدا','عدة','عدد','عدم','عدَّ','عسى','عشر','عشرة','عشرون','عشرين','عل','علق','علم','على','علي','عليك','عليه','عليها','علًّ','عن','عند','عندما','عنه','عنها','عوض','عيانا','عين','عَدَسْ','غ','غادر','غالبا','غدا','غداة','غير','غين','ـ','ف','فإن','فاء','فان','فانه','فبراير','فرادى','فضلا','فقد','فقط','فكان','فلان','فلس','فهو','فو','فوق','فى','في','فيفري','فيه','فيها','ق','قاطبة','قاف','قال','قام','قبل','قد','قرش','قطّ','قلما','قوة','ك','كأن','كأنّ','كأيّ','كأيّن','كاد','كاف','كان','كانت','كانون',
        'كثيرا','كذا','كذلك','كرب','كسا','كل','كلتا','كلم','كلَّا','كلّما','كم','كما','كن','كى','كيت','كيف','كيفما','كِخ','ل','لأن','لا','لا','سيما','لات','لازال','لاسيما','لام','لايزال','لبيك','لدن','لدى','لدي','لذلك','لعل','لعلَّ','لعمر','لقاء','لكن','لكنه','لكنَّ','للامم','لم','لما','لمّا','لن','له','لها','لهذا','لهم','لو','لوكالة','لولا','لوما','ليت','ليرة','ليس','ليسب','م','مئة','مئتان','ما','ما','أفعله','ما','انفك','ما','برح','مائة','ماانفك','مابرح','مادام','ماذا','مارس','مازال','مافتئ','ماي','مايزال','مايو','متى','مثل','مذ','مرّة','مساء','مع','معاذ','معه','معها','مقابل','مكانكم','مكانكما','مكانكنّ',
        'مكانَك','مليار','مليم','مليون','مما','من','منذ','منه','منها','مه','مهما','ميم','ن','نا','نبَّا','نحن','نحو','نعم','نفس','نفسه','نهاية','نوفمبر','نون','نيسان','نيف','نَخْ','نَّ','ه','هؤلاء','ها','هاء','هاكَ','هبّ','هذا','هذه','هل','هللة','هلم','هلّا','هم','هما','همزة','هن','هنا','هناك','هنالك','هو','هي','هيا','هيهات','هيّا','هَؤلاء','هَاتانِ','هَاتَيْنِ','هَاتِه','هَاتِي','هَجْ','هَذا','هَذانِ','هَذَيْنِ','هَذِه','هَذِي','هَيْهات','و','و6','وأبو','وأن','وا','واحد','واضاف','واضافت','واكد','والتي','والذي','وان','واهاً','واو','واوضح','وبين','وثي','وجد','وراءَك','ورد','وعلى','وفي','وقال','وقالت','وقد','وقف','وكان','وكانت',
        'ولا','ولايزال','ولكن','ولم','وله','وليس','ومع','ومن','وهب','وهذا','وهو','وهي','وَيْ','وُشْكَانَ','ى','ي','ياء','يفعلان','يفعلون','يكون','يلي','يمكن','يمين','ين','يناير','يوان','يورو','يوليو','يوم','يونيو','ّأيّان','']

        count1 = 0

        for i in Edited_paragraph:
            querywords = Edited_paragraph[count1].split()
            resultwords  = [word for word in querywords if word not in sss]
            Edited_paragraph[count1] = ' '.join(resultwords)
            count1 = count1 + 1

        ArListem = ArabicLightStemmer()
        stemmedWords = []
        for word in Edited_paragraph:
            #word1 = u'أفتضاربانني'
            # stemming word
            stem = ArListem.light_stem(word)
            # # extract stem
            ArListem.get_stem()
            # # extract root
            stemmedWords.append(ArListem.get_root())

        # text = " ".join(stemmedWords)
        # text
        stemmedWords = stemmedWords[0:-1]

        lemmer = qalsadi.lemmatizer.Lemmatizer()
        counterList = []

        for i in stemmedWords:
            counter = 0
            lemmas = lemmer.lemmatize_text(i, return_pos=True)
            for i in lemmas:
                if i[1] == 'noun':
                    counter +=1
            counterList.append(counter)


        list1 = []

        # X = input("Enter first string: ").lower()
        # Y = input("Enter second string: ").lower()

        for index, X in enumerate(stemmedWords):
            x = index 
            for Y in stemmedWords[index+1:]:
            
                x = x + 1
                l1 = []
                l2 = []
                # tokenization
                X_list = word_tokenize(X)
                Y_list = word_tokenize(Y)

                # remove stop words from the string
                X_set = {w for w in X_list}
                Y_set = {w for w in Y_list}

                # form a set containing keywords of both strings
                rvector = X_set.union(Y_set)

                # create a vector
                for w in rvector:
                    if w in X_set: l1.append(1) 
                    else: l1.append(0)
                    if w in Y_set: l2.append(1)
                    else: l2.append(0)

                # cosine formula
                c = 0
                for i in range(len(rvector)):
                    c+= l1[i]*l2[i]
                if c != 0:
                    list1.append((index,x,{"w":c}))

        
        cosine = c / float((sum(l1)*sum(l2))**0.5)

        def pagerank(G, alpha=0.85, personalization=None,
			max_iter=100, tol=1.0e-6, nstart=None, weight='weight',
			dangling=None):
  
            if len(G) == 0:
                return {}

            if not G.is_directed():
                D = G.to_directed()
            else:
                D = G

            # Create a copy in (right) stochastic form
            W = nx.stochastic_graph(D, weight=weight)
            N = W.number_of_nodes()

            # Choose fixed starting vector if not given
            if nstart is None:
                x = dict.fromkeys(W, 1.0 / N)
            else:
                # Normalized nstart vector
                #s = float(sum(nstart.values()))
                s = 1
                #x = dict((k, v / s) for k, v in nstart.items())
                x = dict((k, v) for k, v in nstart.items())

            if personalization is None:

                # Assign uniform personalization vector if not given
                p = dict.fromkeys(W, 1.0 / N)
            else:
                missing = set(G) - set(personalization)
                if missing:
                    raise NetworkXError('Personalization dictionary '
                                        'must have a value for every node. '
                                        'Missing nodes %s' % missing)
                s = float(sum(personalization.values()))
                p = dict((k, v / s) for k, v in personalization.items())

            if dangling is None:

                # Use personalization vector if dangling vector not specified
                dangling_weights = p
            else:
                missing = set(G) - set(dangling)
                if missing:
                    raise NetworkXError('Dangling node dictionary '
                                        'must have a value for every node. '
                                        'Missing nodes %s' % missing)
                s = float(sum(dangling.values()))
                dangling_weights = dict((k, v/s) for k, v in dangling.items())
            dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

            # power iteration: make up to max_iter iterations
            for _ in range(max_iter):
                xlast = x
                x = dict.fromkeys(xlast.keys(), 0)
                danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
                for n in x:

                    # this matrix multiply looks odd because it is
                    # doing a left multiply x^T=xlast^T*W
                    for nbr in W[n]:
                        x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
                    x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]

                # check convergence, l1 norm
                err = sum([abs(x[n] - xlast[n]) for n in x])
                if err < N*tol:
                    return x
            raise NetworkXError('pagerank: power iteration failed to converge '
                                'in %d iterations.' % max_iter)


        nodesx = [i for i in range(0,len(stemmedWords))]

        Graphx = nx.Graph()
        Graphx.add_nodes_from(nodesx)
        Graphx.add_edges_from(list1)

        Number_of_noun = {index:item for index,item in enumerate(counterList)}

        pr = nx.pagerank(Graphx,0.4,nstart=Number_of_noun)

        marklist=sorted((value, key) for (key,value) in pr.items())
        sortdict=dict([(k,v) for v,k in marklist])

        res = dict(OrderedDict(reversed(list(sortdict.items()))))

        counter = 0
        aa = ''
        for i, k in enumerate(res):
            if i == 0 or i == 1:
                continue
            if counter != 8:
                aa = aa + Original_paragraph[k] + ". "
                counter +=1
            else:
                break

        return render(request, 'index.html', {'text': original_data})
    text = "file should be in .text formate"
    return render(request,"index.html",{"text": 1111111111})
# Create your views here.
