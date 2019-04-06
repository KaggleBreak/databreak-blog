import IndexLayout from '../layouts';
import Wrapper from '../components/Wrapper';
import SiteNav from '../components/header/SiteNav';
import { SiteHeader, outer, inner, SiteMain } from '../styles/shared';
import * as React from 'react';
import { css } from '@emotion/core';

import { PostFullHeader, PostFullTitle, NoImage, PostFull } from '../templates/post';
import { PostFullContent } from '../components/PostContent';
import Footer from '../components/Footer';
import Helmet from 'react-helmet';

const PageTemplate = css`
  .site-main {
    background: #fff;
    padding-bottom: 4vw;
  }
`;


const About: React.FunctionComponent = () => (
  <IndexLayout>
    <Helmet>
      <title>About</title>
    </Helmet>
    <Wrapper css={PageTemplate}>
      <header css={[outer, SiteHeader]}>
        <div css={inner}>
          <SiteNav />
        </div>
      </header>
      <main id="site-main" className="site-main" css={[SiteMain, outer]}>
        <article className="post page" css={[PostFull, NoImage]}>
          <PostFullHeader>
            <PostFullTitle>About</PostFullTitle>
          </PostFullHeader>

          <PostFullContent className="post-full-content">
            <div className="post-content">
              <p>
                '데이터뽀개기'는 2015년에 개설된 캐글(kaggle.com) 플랫폼 대회를 참여하는 스터디 그룹 '캐글뽀개기'에서 출발하였습니다. 처음 라면을 끓이는 사람처럼 데이터를 쉽게 뽀개는 것을 목표로 수평적으로 공부할 수 있는 오픈 모임을 지향합니다. 데이터 관련 현장에서 일하거나, 데이터 분야로 이직/취업 하려는 사람들의 Practical Data Playground를 만들고자 합니다.
              </p>
              <blockquote>
                <p>
                데이터뽀개기 공식 블로그는 데이터뽀개기 내 운영자들이 공동 작성하는 팀 Github 블로그입니다. 
                전 세계에 있는 주요 Data Science 커뮤니티 사이트에 있는 데이터(분석, 엔지니어링, 사이언스) 아티클 자료들을 한글화 하여 데이터뽀개기 회원 분들에게 공유하는 컨트리뷰터가 Creator 운영진의 역할입니다. 
                </p>
              </blockquote>
              <p>
              Kdnuggets, Towards Data Science, Analyticsvidhya, Medium, Kaggle Blog와 같은 사이트를 번역합니다.
                 <a href="https://github.com/KaggleBreak/gatsby-casper">on Github</a>.
              </p>
              <p>
                만약 데이터뽀개기 공식 블로그를 돕거나 컨트리뷰터 하고 싶다면 다음의 이메일 주소 (operator@databreak.org)에 연락해주세요!
              </p>
            </div>
          </PostFullContent>
        </article>
      </main>
      <Footer />
    </Wrapper>
  </IndexLayout>
);

export default About;
