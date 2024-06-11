filename="Faster_Swin_l_w12_DeformRoI_ms_3.pth"
file_id="1B9FPVqvhPGbXgUdS2eRKIUP2M5zhnhES"
query=`curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=${file_id}" \
| perl -nE'say/uc-download-link.*? href="(.*?)\">/' \
| sed -e 's/amp;//g' | sed -n 2p`
url="https://drive.google.com$query"
curl -b ./cookie.txt -L -o ${filename} $url