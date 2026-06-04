from dulwich import porcelain
from dulwich.repo import Repo
import traceback
repo=Repo('.')
try:
    porcelain.push(repo, b'https://github.com/vedantgangaputra/ML_Project_shoppers.git', b'main')
    print('push success')
except Exception as e:
    traceback.print_exc()
