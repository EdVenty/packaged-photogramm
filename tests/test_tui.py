"""Smoke test: TUI against a live API server."""
import asyncio

from test_integration import server, _setup  # noqa: F401  (fixture reuse)


def test_tui_lists_jobs_and_opens_detail(server):  # noqa: F811
    url, _ = server
    api, job = _setup(url, preset="underwater")
    from pgtui.app import PhotogrammApp
    from pgtui.screens import JobScreen, NewJobScreen
    from textual.widgets import DataTable

    async def scenario():
        app = PhotogrammApp(url, "tok", refresh=0.5)
        async with app.run_test(size=(160, 50)) as pilot:
            for _ in range(40):
                await pilot.pause(0.1)
                if app.query_one("#jobs-table", DataTable).row_count:
                    break
            t = app.query_one("#jobs-table", DataTable)
            assert t.row_count == 1
            await pilot.press("n")
            for _ in range(30):
                await pilot.pause(0.1)
                if isinstance(app.screen, NewJobScreen):
                    break
            assert isinstance(app.screen, NewJobScreen)
            payload = app.screen._collect()
            assert payload["preset"] == "standard" and payload["params"]["sfm"]["matcher"] == "exhaustive"
            app.screen.query_one("#preset").value = "underwater-refractive-flat"
            await pilot.pause(0.2)
            payload = app.screen._collect()
            assert payload["params"]["underwater"]["camera"] == "refractive"
            assert payload["params"]["splat"]["method"] == "water-splatting"
            await pilot.press("escape")
            await pilot.pause(0.1)
            t.focus()
            await pilot.press("enter")
            for _ in range(30):
                await pilot.pause(0.1)
                if isinstance(app.screen, JobScreen) and app.screen.job:
                    break
            assert isinstance(app.screen, JobScreen)
            assert app.screen.query_one("#stages", DataTable).row_count == 4  # preprocess, sfm, mvs, splat
            await pilot.press("escape")

    asyncio.run(scenario())
    # job created by payload round-trips through the API
    assert api.job(job["id"])["params"]["underwater"]["enhance"] == "redcomp"
