# gatsby-casper [![Netlify Status](https://api.netlify.com/api/v1/badges/152d2680-84d3-4401-ad1e-fe39d4d046f2/deploy-status)](https://app.netlify.com/sites/databreak/deploys)

This is a static blog generator and starter gatsby repo. A port of [Casper](https://github.com/TryGhost/Casper) v2 a theme from [Ghost](https://ghost.org/) for [GatsbyJS](https://www.gatsbyjs.org/) using [TypeScript](https://www.typescriptlang.org/).

## 설치 및 실행 가이드

```
# install node, typescript (MAC)
$ brew install node
$ npm install -g typescript

# install node, typescript (WINDOWS)
# https://nodejs.org/ko/download/
$ npm install -g typescript

# install npm package
$ npm install
$ npm run dev
```

## 글 작성 후 올리는 방법에 대한 가이드

- [가이드라인](https://github.com/KaggleBreak/gatsby-casper/blob/master/GUIDELINE.md) 문서 참고
- 로컬 환경에서 글 작성 후 빌드를 통해 깨지지 않는지 확인
- 태그, 저자명을 새로 추가해야 하는 경우 `src/content` 경로의 `author.yaml`, `tag.yaml` 확인
