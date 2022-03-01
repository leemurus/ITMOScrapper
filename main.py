from datetime import datetime

from bs4 import BeautifulSoup
import aiohttp
from aiohttp.client_exceptions import (
    ServerDisconnectedError, ClientConnectionError
)
import asyncio
from functools import wraps
import logging
import re
import itertools
from typing import Iterable, Dict
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQ_ID = 0
RESP_ID = 0


class ServerError(Exception):
    pass


class ITMOScrapper:
    BASE_URL = "https://itmo.ru/"
    PERSONALII_POSTFIX = "/ru/personlist/personalii.htm"

    ACADEMIC_DEGREE = "ученая степень:"
    ACADEMIC_POSITION = "должность:"

    RETRY_NUMBER = 20

    def __init__(self, session: aiohttp.ClientSession):
        self._session = session

    def _retry(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            for retry_id in range(ITMOScrapper.RETRY_NUMBER):
                try:
                    result = await func(self, *args, **kwargs)
                except (
                        ServerDisconnectedError,
                        ServerError,
                        ClientConnectionError,
                ) as e:
                    if retry_id == ITMOScrapper.RETRY_NUMBER - 1:
                        raise e

                    logger.warning(
                        f"Retry request with arguments {args}, {kwargs}"
                    )
                else:
                    return result

        return wrapper

    async def _do_request2(self, postfix: str) -> str:
        url = urljoin(self.BASE_URL, postfix)
        async with self._session.get(url) as resp:
            global REQ_ID
            REQ_ID += 1
            logger.info(f"REQ_ID {REQ_ID}")
            text = await resp.text()
            return text

    async def _do_request(self, postfix: str) -> str:
        url = urljoin(self.BASE_URL, postfix)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                global REQ_ID
                REQ_ID += 1
                # logger.info(f"REQ_ID {REQ_ID}")
                text = await resp.text()
                return text

    @_retry
    async def fetch_letter_urls(self) -> Iterable[str]:
        text = await self._do_request(self.PERSONALII_POSTFIX)
        soup = BeautifulSoup(text, features="html.parser")
        objs = soup.select(".b-page-nav li a")
        links = (obj.get("href") for obj in objs)
        return links

    async def fetch_all_teachers(self, urls) -> Iterable[str]:
        results = await asyncio.gather(*[
            self.fetch_teachers_by_url(url) for url in urls
        ])

        urls = set()
        for result in results:
            urls.update(result)

        return urls

    @_retry
    async def fetch_teachers_by_url(self, url) -> Iterable[str]:
        text = await self._do_request(url)
        soup = BeautifulSoup(text, features="html.parser")

        if not soup.find():
            raise ServerError("Server error")

        objs = soup.select(".main-content .phoneList article a")
        links = list(obj.get("href") for obj in objs)

        return links

    async def fetch_all_teachers_details(self, urls):
        return list(await asyncio.gather(*[
            self.fetch_teacher_details(url) for url in urls
        ]))

    @_retry
    async def fetch_teacher_details(self, url) -> Dict[str, any]:
        text = await self._do_request(url)

        soup = BeautifulSoup(text, features="html.parser")
        if not soup.find():
            raise ServerError("Server error")

        degrees = await self._get_academic_degrees(soup)
        logger.info(f"Degrees: {url} {degrees}")

        positions = await self._get_academic_positions(soup)
        logger.info(f"Positions: {url} {positions}")

        publications_number = await self._get_publications_number(soup)
        logger.info(publications_number)

        return {
            "url": url,
            "degrees": degrees,
            "positions": positions,
            "publications_number": publications_number,
        }

    async def _get_academic_degrees(
            self, soup: BeautifulSoup,
    ) -> Iterable[str]:
        info = soup.select(".c-personCard-details dl")[0]

        degrees = []
        for line in info:
            if line.text.strip().lower() == self.ACADEMIC_DEGREE:
                degree = line.findNext("dd").text.strip().lower()
                degrees.append(re.sub(" +", " ", degree))

        return degrees

    async def _get_academic_positions(
            self, soup: BeautifulSoup,
    ) -> Iterable[str]:
        info = soup.select(".c-personCard-details dl")[1]

        positions = []

        for line in info:
            if line.text.strip().lower() == self.ACADEMIC_POSITION:
                cur_position = []
                for inner_line in line.findNext("dd").contents:
                    if inner_line.name == "br":
                        positions.append(" ".join(cur_position))
                        cur_position = []
                        continue

                    cur_position.append(inner_line.text.strip())

                if cur_position:
                    positions.append(" ".join(cur_position))

        return positions

    async def _get_publications_number(self, soup: BeautifulSoup) -> int:
        info = soup.select_one(".nav-tabs a:contains(Публикации) .badge")
        return int(info.text) if info else 0


async def main():
    conn = aiohttp.TCPConnector(limit=500, ttl_dns_cache=300)

    async with aiohttp.ClientSession(connector=conn,
                                     raise_for_status=True) as session:
        scrapper = ITMOScrapper(session)
        letter_urls = list(await scrapper.fetch_letter_urls())
        logger.info(f"Letters: {len(letter_urls)}")

        teachers_urls = list(await scrapper.fetch_all_teachers(letter_urls))
        logger.info(f"Teachers number: {len(teachers_urls)}")

        users = await scrapper.fetch_all_teachers_details(teachers_urls)

        logger.info('1. Employees with multiple positions:')
        logger.info([
            user['url'] for user in users if len(user['positions']) >= 2
        ])

        logger.info('2. Maximum number of positions for an employee:')
        logger.info(max(len(user["positions"]) for user in users))

        logger.info('3. Number of employees with academic degrees:')
        logger.info(sum(1 for user in users if user['degrees']))

        logger.info('4. Academic degrees:')
        logger.info(
            len(set(itertools.chain(*(user['degrees'] for user in users))))
        )

        logger.info('5. Chart of publications from teachers:')
        from collections import Counter
        import matplotlib.pyplot as plt
        import numpy as np

        counter = Counter(user['publications_number'] for user in users)
        x, y = np.hsplit(
            np.array(sorted(list(counter.items()), key=lambda x: x[0])), 2)

        plt.plot(x, y)
        plt.show()

        logger.info('6. Average publications number')
        with_degree = sum(1 for user in users if user['degrees'])
        publications_number_with_degr = sum(
            user['publications_number']
            for user in users
            if user['degrees']
        )
        without_degree = sum(1 for user in users if not user['degrees'])
        publications_number_without_degr = sum(
            user['publications_number']
            for user in users
            if not user['degrees']
        )
        avg_with_degr = publications_number_with_degr / with_degree
        avg_without_degr = publications_number_without_degr / without_degree
        logger.info(f'With a scientific degree: {avg_with_degr}')
        logger.info(f'Without a scientific degree: {avg_without_degr}')


if __name__ == "__main__":
    start_time = datetime.now()
    asyncio.run(main())
    logger.warning(f"Result: {datetime.now() - start_time}")
