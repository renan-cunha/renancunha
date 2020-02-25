snap run hugo
cd public
git add .
rm 404.html
git commit -m "Build website"
git push origin master

