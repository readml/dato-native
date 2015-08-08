from bs4 import BeautifulSoup as bs
import os, sys, logging, string, glob
import cssutils as cu
import json
import ipdb
ferr = open("errors_in_scraping.log","w")

def parse_page(in_file, urlid):
    """ parameters:
            - in_file: file to read raw_data from
            - url_id: id of each page from file_name """
    page = open(in_file, 'r')
    soup = bs(page)
    doc = {
            "id": urlid, 
            "text":parse_text(soup),
            "title":parse_title(soup ),
            "links":parse_links(soup),
            "images":parse_images(soup),
           }

    return doc

def parse_text(soup):
    """ parameters:
            - soup: beautifulSoup4 parsed html page
        out:
            - textdata: a list of parsed text output by looping over html paragraph tags
        note:
            - could soup.get_text() instead but the output is more noisy """
    textdata = ['']

    for text in soup.find_all('p'):
        try:
            textdata.append(text.text.encode('ascii','ignore').strip())
        except Exception:
            continue

    return filter(None,textdata)

def parse_title(soup):
    """ parameters:
            - soup: beautifulSoup4 parsed html page
        out:
            - title: parsed title """

    title = ['']

    try:
        title.append(soup.title.string.encode('ascii','ignore').strip())
    except Exception:
        return title

    return filter(None,title)

def parse_links(soup):
    """ parameters:
            - soup: beautifulSoup4 parsed html page
        out:
            - linkdata: a list of parsed links by looping over html link tags
        note:
            - some bad links in here, this could use more processing """

    linkdata = ['']

    for link in soup.find_all('a'):
        try:
            linkdata.append(str(link.get('href').encode('ascii','ignore')))
        except Exception:
            continue

    return filter(None,linkdata)


def parse_images(soup):
    """ parameters:
            - soup: beautifulSoup4 parsed html page
        out:
            - imagesdata: a list of parsed image names by looping over html img tags """
    imagesdata = ['']

    for image in soup.findAll("img"):
        try:
            imagesdata.append("%(src)s"%image)
        except Exception:
            continue

    return filter(None,imagesdata)


def main(argv):
    """ parameters:
                - argv: sys args from the command line that consist of:
                             <input_raw_dir> <output_directory>
                * input_raw_dir: directory to read raw input html files
                * output_directory: directory to save processed html files

        note:
                - this will loop over all raw_files and create processed ouput for
                  a give site_id IF input data for that id exists, otherwise it will
                  skip it """
    if len(argv) < 3:

        print " Usage: python process_html.py <input_raw_dir> <output_directory>"
        return

    else:

        inFolder = argv[1]
        outputDirectory = argv[2]

        if not os.path.exists(inFolder):
            print inFolder," does not exist"
            return

        if not os.path.exists(outputDirectory):
            os.makedirs(outputDirectory)

        cu.log.setLevel(logging.CRITICAL)
	json_array, last_bucket = [], str(0)

        fIn = glob.glob( inFolder + '/*/*raw*')
        for idx, filename in enumerate(fIn):

            if idx % 1000 == 0:
                print "Processed %d HTML files" % idx

            filenameDetails = filename.split("/")
            urlId = filenameDetails[-1].split('_')[0]
            bucket = filenameDetails[-2]
            
	    # PROBLEM SAVING BUCKETS:
	    # because of how this is written, the last bucket will be parsed, but not saved.
	    # for instance, if you had one bucket in the directory, that bucket would not be saved.
	    # if you had 3 buckets in the directory, the first 2 would be saved but last wouldn't
   	    # TEMPFIX: put a ipdb.set_trace() at the end, in case the json array is not saved
	    if bucket != last_bucket:            
                print 'SAVING BUCKET %s' % last_bucket
                out_file = os.path.join(outputDirectory, 'chunk' + last_bucket + '.json')

                with open(out_file, mode='w') as feedsjson:
                    for entry in json_array:
                        json.dump(entry, feedsjson)
                        feedsjson.write('\n')

                feedsjson.close()
                json_array = []  
                last_bucket = bucket 

            try:
                doc = parse_page(filename, urlId)
            except Exception as e:
                ferr.write("parse error with reason : "+str(e)+" on page "+urlId+"\n")
                continue

            json_array.append(doc)
	import ipdb; ipdb.set_trace(); 
           
    print "Scraping completed .. There may be errors .. check log at errors_in_scraping.log"

if __name__ == "__main__":
   main(sys.argv)

