# databreak-blog [![Netlify Status](https://api.netlify.com/api/v1/badges/dfb8d7e0-be83-40ca-9019-e8806ef4fb75/deploy-status)](https://app.netlify.com/sites/databreak/deploys)

This is a static blog generator gatsby repo. A port of [Casper](https://github.com/TryGhost/Casper) v2 a theme from [Ghost](https://ghost.org/) for [GatsbyJS](https://www.gatsbyjs.org/) using [TypeScript](https://www.typescriptlang.org/).

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
- 컨텐츠 이미지 사이즈는 width 1080, height 700 을 권장
- 프로필 이미지 사이즈는 400px 이상을 권장


## 데이터뽀개기 블로그 월별 진행상황 체크(테이블)

- 상태 표시 : 완료(페이스북 공유까지), 진행중(블로그 글 작성까지), 미완료(이슈 발행까지)


| 이름   | 3월 | 4월 | 5월 | 6월 | 7월 | 8월 | 9월 | 10월 | 11월 |
|--------|-----|-----|-----|-----|-----|-----|------|------|------|
| 이상열 |완료|완료|진행중|     |     |     |      |      |      |
| 박준영 |블로그 개발|진행중|     |     |     |     |      |      |      |
| 이규영 |합류전|완료|진행중|     |     |     |      |      |      |
| 류영표 |완료|진행중|     |     |     |     |      |      |      |
| 조영승 |완료|진행중|     |     |     |     |      |      |      |
| 김강민 |합류전|완료|     |     |     |     |      |      |      |
| 전인아 |완료|진행중|     |     |     |     |      |      |      |
| 김현우 |완료|완료|진행중|     |     |     |      |      |      |
| 조규원 |완료|완료|진행중|     |     |     |      |      |      |
