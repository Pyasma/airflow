# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.engine.url import make_url

from airflow.models import Connection
from airflow.providers.apache.impala.hooks.impala import ImpalaHook

DEFAULT_CONN_ID = "impala_default"
DEFAULT_HOST = "impala-server.example.com"
DEFAULT_PORT = 21050
DEFAULT_LOGIN = "test_user"
DEFAULT_PASSWORD = "test_password"


@pytest.fixture
def impala_hook_fixture() -> ImpalaHook:
    """Fixture for ImpalaHook with mocked cursor operations."""
    hook = ImpalaHook()
    mock_get_conn = MagicMock()
    mock_get_conn.return_value.cursor = MagicMock()
    mock_get_conn.return_value.cursor.return_value.rowcount = 2
    hook.get_conn = mock_get_conn  # type:ignore[method-assign]

    return hook


@pytest.fixture
def impala_hook() -> ImpalaHook:
    """Fixture for basic ImpalaHook instance."""
    return ImpalaHook()


@patch("airflow.providers.apache.impala.hooks.impala.connect", autospec=True)
def test_get_conn(mock_connect):
    """Test get_conn establishes connection with correct parameters."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="login",
            password="password",
            host="host",
            port=21050,
            schema="test",
            extra={"use_ssl": True},
        )
    )
    hook.get_conn()
    mock_connect.assert_called_once_with(
        host="host", port=21050, user="login", password="password", database="test", use_ssl=True
    )


@patch("airflow.providers.apache.impala.hooks.impala.connect", autospec=True)
def test_get_conn_kerberos(mock_connect):
    """Test get_conn with Kerberos authentication."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="login",
            password="password",
            host="host",
            port=21050,
            schema="test",
            extra={"auth_mechanism": "GSSAPI", "use_ssl": True},
        )
    )
    hook.get_conn()
    mock_connect.assert_called_once_with(
        host="host",
        port=21050,
        user="login",
        password="password",
        database="test",
        use_ssl=True,
        auth_mechanism="GSSAPI",
    )


@patch("airflow.providers.common.sql.hooks.sql.DbApiHook.insert_rows")
def test_insert_rows(mock_insert_rows, impala_hook_fixture):
    """Test insert_rows delegates to parent class correctly."""
    table = "table"
    rows = [("hello",), ("world",)]
    target_fields = None
    commit_every = 10
    impala_hook_fixture.insert_rows(table, rows, target_fields, commit_every)
    mock_insert_rows.assert_called_once_with(table, rows, None, 10)


def test_get_first_record(impala_hook_fixture):
    """Test get_first returns the first record from query results."""
    statement = "SQL"
    result_sets = [("row1",), ("row2",)]
    impala_hook_fixture.get_conn.return_value.cursor.return_value.fetchone.return_value = result_sets[0]

    assert result_sets[0] == impala_hook_fixture.get_first(statement)
    impala_hook_fixture.get_conn.return_value.cursor.return_value.execute.assert_called_once_with(statement)


def test_get_records(impala_hook_fixture):
    """Test get_records returns all records from query results."""
    statement = "SQL"
    result_sets = [("row1",), ("row2",)]
    impala_hook_fixture.get_conn.return_value.cursor.return_value.fetchall.return_value = result_sets

    assert result_sets == impala_hook_fixture.get_records(statement)
    impala_hook_fixture.get_conn.return_value.cursor.return_value.execute.assert_called_once_with(statement)


def test_get_df(impala_hook_fixture):
    """Test get_df returns pandas DataFrame with correct structure."""
    statement = "SQL"
    column = "col"
    result_sets = [("row1",), ("row2",)]
    impala_hook_fixture.get_conn.return_value.cursor.return_value.description = [(column,)]
    impala_hook_fixture.get_conn.return_value.cursor.return_value.fetchall.return_value = result_sets
    df = impala_hook_fixture.get_df(statement, df_type="pandas")

    assert column == df.columns[0]

    assert result_sets[0][0] == df.values.tolist()[0][0]
    assert result_sets[1][0] == df.values.tolist()[1][0]

    impala_hook_fixture.get_conn.return_value.cursor.return_value.execute.assert_called_once_with(statement)


def test_get_df_polars(impala_hook_fixture):
    """Test get_df returns Polars DataFrame with correct structure."""
    statement = "SQL"
    column = "col"
    result_sets = [("row1",), ("row2",)]
    mock_execute = MagicMock()
    mock_execute.description = [(column, None, None, None, None, None, None)]
    mock_execute.fetchall.return_value = result_sets
    impala_hook_fixture.get_conn.return_value.cursor.return_value.execute.return_value = mock_execute

    df = impala_hook_fixture.get_df(statement, df_type="polars")
    assert column == df.columns[0]
    assert result_sets[0][0] == df.row(0)[0]
    assert result_sets[1][0] == df.row(1)[0]


# SQLAlchemy URL Property Tests


@pytest.mark.parametrize(
    "login, password, host, port, schema, extra, expected_password, expected_port, expected_database, expected_query",
    [
        pytest.param(
            "test_user",
            "test_password",
            "impala-server.example.com",
            21050,
            "test_database",
            {"use_ssl": True, "auth_mechanism": "PLAIN"},
            "test_password",
            21050,
            "test_database",
            {"use_ssl": True, "auth_mechanism": "PLAIN"},
            id="all_fields_provided",
        ),
        pytest.param(
            "minimal_user",
            None,
            "minimal-host",
            None,
            None,
            None,
            "",
            21050,
            None,
            {},
            id="minimal_required_fields",
        ),
        pytest.param(
            "user",
            "pass",
            "host",
            21000,
            None,
            None,
            "pass",
            21000,
            None,
            {},
            id="custom_port",
        ),
        pytest.param(
            "user",
            None,
            "host",
            None,
            "production_db",
            None,
            "",
            21050,
            "production_db",
            {},
            id="with_schema",
        ),
        pytest.param(
            "user",
            "",
            "host",
            None,
            None,
            {},
            "",
            21050,
            None,
            {},
            id="empty_password_and_extras",
        ),
    ],
)
def test_sqlalchemy_url_property(
    impala_hook,
    login,
    password,
    host,
    port,
    schema,
    extra,
    expected_password,
    expected_port,
    expected_database,
    expected_query,
):
    """Test sqlalchemy_url property with various connection configurations."""
    mock_conn = Connection(
        login=login,
        password=password,
        host=host,
        port=port,
        schema=schema,
        extra=extra,
    )

    with patch.object(impala_hook, "get_connection", return_value=mock_conn):
        url = impala_hook.sqlalchemy_url

        assert url.drivername == "impala"
        assert url.username == login
        assert url.password == expected_password
        assert url.host == str(host)
        assert url.port == expected_port
        assert url.database == expected_database
        assert url.query == expected_query


@pytest.mark.parametrize(
    "extra, expected_query",
    [
        pytest.param(
            {"use_ssl": True, "timeout": None, "auth_mechanism": "GSSAPI"},
            {"use_ssl": True, "auth_mechanism": "GSSAPI"},
            id="filters_none_values",
        ),
        pytest.param(
            {"use_ssl": True, "__extra__": {"some": "value"}},
            {"use_ssl": True},
            id="filters_dunder_extra",
        ),
        pytest.param(
            {
                "use_ssl": True,
                "auth_mechanism": "GSSAPI",
                "kerberos_service_name": "impala",
                "timeout": 300,
            },
            {
                "use_ssl": True,
                "auth_mechanism": "GSSAPI",
                "kerberos_service_name": "impala",
                "timeout": 300,
            },
            id="multiple_extras",
        ),
    ],
)
def test_sqlalchemy_url_extras_handling(impala_hook, extra, expected_query):
    """Test sqlalchemy_url property correctly handles extra parameters."""
    mock_conn = Connection(
        login="user",
        host="host",
        extra=extra,
    )

    with patch.object(impala_hook, "get_connection", return_value=mock_conn):
        url = impala_hook.sqlalchemy_url

        assert url.query == expected_query
        for key in expected_query:
            assert key in url.query


def test_sqlalchemy_url_when_extra_dejson_is_none(impala_hook):
    """Test sqlalchemy_url property when extra_desjson returns None."""
    mock_conn = MagicMock(spec=Connection)
    mock_conn.login = "user"
    mock_conn.host = "host"
    mock_conn.password = None
    mock_conn.port = None
    mock_conn.schema = None
    mock_conn.extra_desjson = None

    with patch.object(impala_hook, "get_connection", return_value=mock_conn):
        url = impala_hook.sqlalchemy_url

        assert url.query == {}


@pytest.mark.parametrize(
    "login, host, expected_error",
    [
        pytest.param(
            "user",
            None,
            "Impala Connection Error: 'host' is missing in the connection",
            id="host_is_none",
        ),
        pytest.param(
            None,
            "host",
            "Impala Connection Error: 'login' is missing in the connection",
            id="login_is_none",
        ),
        pytest.param(
            None,
            None,
            "Impala Connection Error: '(host|login)' is missing in the connection",
            id="both_host_and_login_none",
        ),
    ],
)
def test_sqlalchemy_url_validation_errors(impala_hook, login, host, expected_error):
    """Test sqlalchemy_url property raises ValueError for missing required fields."""
    mock_conn = Connection(
        login=login,
        host=host,
    )

    with patch.object(impala_hook, "get_connection", return_value=mock_conn):
        with pytest.raises(ValueError, match=expected_error):
            _ = impala_hook.sqlalchemy_url


def test_sqlalchemy_url_converts_host_to_string(impala_hook):
    """Test sqlalchemy_url property converts host to string."""
    mock_conn = Connection(
        login="user",
        host="192.168.1.100",
    )

    with patch.object(impala_hook, "get_connection", return_value=mock_conn):
        url = impala_hook.sqlalchemy_url

        assert url.host == "192.168.1.100"
        assert isinstance(url.host, str)


# get_uri() Method Tests


def test_get_uri_returns_string(impala_hook):
    """Test get_uri returns string representation of connection URI."""
    mock_conn = Connection(
        login="user",
        password="secret",
        host="host",
        port=21050,
        schema="db",
    )

    with patch.object(impala_hook, "get_connection", return_value=mock_conn):
        result = impala_hook.get_uri()

        assert isinstance(result, str)
        assert result.startswith("impala://")
        assert "user" in result
        assert "host" in result


def test_get_uri_password_visible(impala_hook):
    """Test get_uri includes password visibly (hide_password=False)."""
    mock_conn = Connection(
        login="user",
        password="my_secret_password",
        host="host",
    )

    with patch.object(impala_hook, "get_connection", return_value=mock_conn):
        result = impala_hook.get_uri()

        assert "my_secret_password" in result
        assert "impala://user:my_secret_password@host:21050" in result


@pytest.mark.parametrize(
    "password, expected_substring",
    [
        pytest.param("", "user:@host", id="empty_password"),
        pytest.param(None, ":@host", id="none_password"),
    ],
)
def test_get_uri_password_formats(impala_hook, password, expected_substring):
    """Test get_uri handles empty and None passwords correctly."""
    mock_conn = Connection(
        login="user",
        password=password,
        host="host",
    )

    with patch.object(impala_hook, "get_connection", return_value=mock_conn):
        result = impala_hook.get_uri()

        assert expected_substring in result


def test_get_uri_with_schema(impala_hook):
    """Test get_uri includes schema in URI path."""
    mock_conn = Connection(
        login="user",
        host="host",
        schema="production",
    )

    with patch.object(impala_hook, "get_connection", return_value=mock_conn):
        result = impala_hook.get_uri()

        assert "/production" in result


def test_get_uri_with_extras_as_query_parameters(impala_hook):
    """Test get_uri includes extras as query parameters."""
    mock_conn = Connection(
        login="user",
        host="host",
        extra={"use_ssl": True, "auth_mechanism": "GSSAPI"},
    )

    with patch.object(impala_hook, "get_connection", return_value=mock_conn):
        result = impala_hook.get_uri()

        assert "?" in result
        assert "use_ssl" in result.lower()
        assert "auth_mechanism" in result.lower() or "GSSAPI" in result


def test_get_uri_complete_example(impala_hook):
    """Test get_uri returns complete URI with all components."""
    mock_conn = Connection(
        login="admin_user",
        password="admin_pass",
        host="impala.example.com",
        port=21000,
        schema="analytics",
        extra={"use_ssl": True},
    )

    expected_uri = "impala://admin_user:admin_pass@impala.example.com:21000/analytics?use_ssl=True"

    with patch.object(impala_hook, "get_connection", return_value=mock_conn):
        uri = impala_hook.get_uri()

        parsed = make_url(uri)
        expected = make_url(expected_uri)

        assert parsed.drivername == expected.drivername
        assert parsed.username == expected.username
        assert parsed.password == expected.password
        assert parsed.host == expected.host
        assert parsed.port == expected.port
        assert parsed.database == expected.database


def test_get_uri_validates_required_fields(impala_hook):
    """Test get_uri raises ValueError when required fields are missing."""
    mock_conn = Connection(
        login="user",
        host=None,
    )

    with patch.object(impala_hook, "get_connection", return_value=mock_conn):
        with pytest.raises(ValueError, match="Impala Connection Error: 'host' is missing in the connection"):
            _ = impala_hook.get_uri()


# Integration & Type Validation Tests


def test_sqlalchemy_url_returns_url_object_type(impala_hook):
    """Test sqlalchemy_url property returns SQLAlchemy URL object."""
    from sqlalchemy.engine import URL

    mock_conn = Connection(
        login="user",
        host="host",
    )

    with patch.object(impala_hook, "get_connection", return_value=mock_conn):
        result = impala_hook.sqlalchemy_url

        assert isinstance(result, URL)


def test_get_uri_calls_sqlalchemy_url_property(impala_hook):
    """Test get_uri delegates to sqlalchemy_url property."""
    mock_conn = Connection(
        login="user",
        host="host",
    )

    with patch.object(impala_hook, "get_connection", return_value=mock_conn):
        url_result = impala_hook.sqlalchemy_url
        uri_result = impala_hook.get_uri()

        assert uri_result == url_result.render_as_string(hide_password=False)


def test_multiple_calls_to_sqlalchemy_url_consistent(impala_hook):
    """Test sqlalchemy_url property returns consistent results across multiple calls."""
    mock_conn = Connection(
        login="user",
        password="pass",
        host="host",
        port=21050,
        schema="db",
    )

    with patch.object(impala_hook, "get_connection", return_value=mock_conn):
        url1 = impala_hook.sqlalchemy_url
        url2 = impala_hook.sqlalchemy_url

        assert str(url1) == str(url2)
        assert url1.username == url2.username
        assert url1.password == url2.password
        assert url1.host == url2.host
        assert url1.port == url2.port
        assert url1.database == url2.database
