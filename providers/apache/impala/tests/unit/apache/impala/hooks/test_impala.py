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
from sqlalchemy.engine import URL

from airflow.models import Connection
from airflow.providers.apache.impala.hooks.impala import ImpalaHook


@pytest.fixture
def impala_hook_fixture() -> ImpalaHook:
    hook = ImpalaHook()
    mock_get_conn = MagicMock()
    mock_get_conn.return_value.cursor = MagicMock()
    mock_get_conn.return_value.cursor.return_value.rowcount = 2
    hook.get_conn = mock_get_conn  # type:ignore[method-assign]

    return hook


@patch("airflow.providers.apache.impala.hooks.impala.connect", autospec=True)
def test_get_conn(mock_connect):
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
    table = "table"
    rows = [("hello",), ("world",)]
    target_fields = None
    commit_every = 10
    impala_hook_fixture.insert_rows(table, rows, target_fields, commit_every)
    mock_insert_rows.assert_called_once_with(table, rows, None, 10)


def test_get_first_record(impala_hook_fixture):
    statement = "SQL"
    result_sets = [("row1",), ("row2",)]
    impala_hook_fixture.get_conn.return_value.cursor.return_value.fetchone.return_value = result_sets[0]

    assert result_sets[0] == impala_hook_fixture.get_first(statement)
    impala_hook_fixture.get_conn.return_value.cursor.return_value.execute.assert_called_once_with(statement)


def test_get_records(impala_hook_fixture):
    statement = "SQL"
    result_sets = [("row1",), ("row2",)]
    impala_hook_fixture.get_conn.return_value.cursor.return_value.fetchall.return_value = result_sets

    assert result_sets == impala_hook_fixture.get_records(statement)
    impala_hook_fixture.get_conn.return_value.cursor.return_value.execute.assert_called_once_with(statement)


def test_get_df(impala_hook_fixture):
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


# SQLAlchemy URL Property Tests - Happy Paths


def test_sqlalchemy_url_with_all_fields():
    """Verify sqlalchemy_url with complete connection configuration."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="test_user",
            password="test_password",
            host="impala-server.example.com",
            port=21050,
            schema="test_database",
            extra={"use_ssl": True, "auth_mechanism": "PLAIN"},
        )
    )
    result = hook.sqlalchemy_url

    assert isinstance(result, URL)
    assert result.drivername == "impala"
    assert result.username == "test_user"
    assert result.password == "test_password"
    assert result.host == "impala-server.example.com"
    assert result.port == 21050
    assert result.database == "test_database"
    assert result.query == {"use_ssl": True, "auth_mechanism": "PLAIN"}


def test_sqlalchemy_url_minimal_required_fields():
    """Verify URL creation with only required fields (host, login)."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="minimal_user",
            host="minimal-host",
        )
    )
    result = hook.sqlalchemy_url

    assert result.host == "minimal-host"
    assert result.username == "minimal_user"
    assert result.password == ""
    assert result.port == 21050
    assert result.database is None
    assert result.query == {}


def test_sqlalchemy_url_with_custom_port():
    """Verify custom port handling."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            password="pass",
            host="host",
            port=21000,
        )
    )
    result = hook.sqlalchemy_url

    assert result.port == 21000


def test_sqlalchemy_url_without_schema():
    """Verify URL creation without database/schema."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            host="host",
        )
    )
    result = hook.sqlalchemy_url

    assert result.database is None


def test_sqlalchemy_url_with_schema():
    """Verify schema is properly mapped to database."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            host="host",
            schema="production_db",
        )
    )
    result = hook.sqlalchemy_url

    assert result.database == "production_db"


def test_sqlalchemy_url_with_empty_password():
    """Verify empty string password handling."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            host="host",
            password="",
        )
    )
    result = hook.sqlalchemy_url

    assert result.password == ""


# SQLAlchemy URL Property Tests - Extras Handling


def test_sqlalchemy_url_filters_none_values_in_extras():
    """Verify None values in extras are filtered out."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            host="host",
            extra={"use_ssl": True, "timeout": None, "auth_mechanism": "GSSAPI"},
        )
    )
    result = hook.sqlalchemy_url

    assert result.query == {"use_ssl": True, "auth_mechanism": "GSSAPI"}
    assert "timeout" not in result.query


def test_sqlalchemy_url_filters_dunder_extra():
    """Verify __extra__ key is filtered from query parameters."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            host="host",
            extra={"use_ssl": True, "__extra__": {"some": "value"}},
        )
    )
    result = hook.sqlalchemy_url

    assert result.query == {"use_ssl": True}
    assert "__extra__" not in result.query


def test_sqlalchemy_url_with_multiple_extras():
    """Verify multiple extra parameters in query."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            host="host",
            extra={
                "use_ssl": True,
                "auth_mechanism": "GSSAPI",
                "kerberos_service_name": "impala",
                "timeout": 300,
            },
        )
    )
    result = hook.sqlalchemy_url

    assert len(result.query) == 4
    assert result.query["use_ssl"] is True
    assert result.query["auth_mechanism"] == "GSSAPI"
    assert result.query["kerberos_service_name"] == "impala"
    assert result.query["timeout"] == 300


def test_sqlalchemy_url_with_empty_extras():
    """Verify empty extras dict handling."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            host="host",
            extra={},
        )
    )
    result = hook.sqlalchemy_url

    assert result.query == {}


def test_sqlalchemy_url_when_extra_dejson_is_none():
    """Verify handling when extra_desjson returns None."""
    hook = ImpalaHook()
    mock_conn = MagicMock(spec=Connection)
    mock_conn.login = "user"
    mock_conn.host = "host"
    mock_conn.password = None
    mock_conn.port = None
    mock_conn.schema = None
    mock_conn.extra_desjson = None
    hook.get_connection = MagicMock(return_value=mock_conn)

    result = hook.sqlalchemy_url

    assert result.query == {}


# SQLAlchemy URL Property Tests - Validation


def test_sqlalchemy_url_raises_error_when_host_is_none():
    """Verify ValueError when host is None."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            host=None,
        )
    )

    with pytest.raises(ValueError, match="Impala Connection Error: 'host' is missing in the connection"):
        _ = hook.sqlalchemy_url


def test_sqlalchemy_url_raises_error_when_login_is_none():
    """Verify ValueError when login is None."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login=None,
            host="host",
        )
    )

    with pytest.raises(ValueError, match="Impala Connection Error: 'login' is missing in the connection"):
        _ = hook.sqlalchemy_url


def test_sqlalchemy_url_raises_error_when_both_host_and_login_none():
    """Verify error when both required fields missing."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login=None,
            host=None,
        )
    )

    with pytest.raises(ValueError, match="Impala Connection Error: '(host|login)' is missing in the connection"):
        _ = hook.sqlalchemy_url


def test_sqlalchemy_url_converts_host_to_string():
    """Verify host is converted to string."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            host="192.168.1.100",
        )
    )
    result = hook.sqlalchemy_url

    assert result.host == "192.168.1.100"
    assert isinstance(result.host, str)


# get_uri() Method Tests


def test_get_uri_returns_string():
    """Verify get_uri returns string representation."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            password="secret",
            host="host",
            port=21050,
            schema="db",
        )
    )
    result = hook.get_uri()

    assert isinstance(result, str)
    assert result.startswith("impala://")
    assert "user" in result
    assert "host" in result


def test_get_uri_password_visible():
    """Verify password is visible in URI (hide_password=False in code)."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            password="my_secret_password",
            host="host",
        )
    )
    result = hook.get_uri()

    assert "my_secret_password" in result
    assert "impala://user:my_secret_password@host:21050" in result


def test_get_uri_with_empty_password():
    """Verify URI format when password is empty."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            password="",
            host="host",
        )
    )
    result = hook.get_uri()

    assert "user:@host" in result


def test_get_uri_with_no_password():
    """Verify URI when password is None."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            password=None,
            host="host",
        )
    )
    result = hook.get_uri()

    assert ":@host" in result


def test_get_uri_with_schema():
    """Verify database appears in URI."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            host="host",
            schema="production",
        )
    )
    result = hook.get_uri()

    assert "/production" in result


def test_get_uri_with_extras_as_query_parameters():
    """Verify extras appear as query parameters in URI."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            host="host",
            extra={"use_ssl": True, "auth_mechanism": "GSSAPI"},
        )
    )
    result = hook.get_uri()

    assert "?" in result
    assert "use_ssl" in result.lower()
    assert "auth_mechanism" in result.lower() or "GSSAPI" in result


def test_get_uri_complete_example():
    """Verify complete URI with all components."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="admin_user",
            password="admin_pass",
            host="impala.example.com",
            port=21000,
            schema="analytics",
            extra={"use_ssl": True},
        )
    )
    result = hook.get_uri()

    assert "impala://" in result
    assert "admin_user" in result
    assert "admin_pass" in result
    assert "impala.example.com" in result
    assert "21000" in result
    assert "analytics" in result
    assert "use_ssl" in result.lower()


def test_get_uri_validates_required_fields():
    """Verify get_uri raises same validation errors as sqlalchemy_url."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            host=None,
        )
    )

    with pytest.raises(ValueError, match="Impala Connection Error: 'host' is missing in the connection"):
        _ = hook.get_uri()


# Integration & Type Validation Tests


def test_sqlalchemy_url_returns_url_object_type():
    """Verify return type is sqlalchemy.engine.URL."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            host="host",
        )
    )
    result = hook.sqlalchemy_url

    assert isinstance(result, URL)


def test_get_uri_calls_sqlalchemy_url_property():
    """Verify get_uri delegates to sqlalchemy_url."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            host="host",
        )
    )
    
    # Call both and verify consistency
    url_result = hook.sqlalchemy_url
    uri_result = hook.get_uri()
    
    # The URI should be the string representation of the URL
    assert uri_result == url_result.render_as_string(hide_password=False)


def test_multiple_calls_to_sqlalchemy_url_consistent():
    """Verify property returns consistent URLs on multiple accesses."""
    hook = ImpalaHook()
    hook.get_connection = MagicMock(
        return_value=Connection(
            login="user",
            password="pass",
            host="host",
            port=21050,
            schema="db",
        )
    )
    
    url1 = hook.sqlalchemy_url
    url2 = hook.sqlalchemy_url
    
    assert str(url1) == str(url2)
    assert url1.username == url2.username
    assert url1.password == url2.password
    assert url1.host == url2.host
    assert url1.port == url2.port
    assert url1.database == url2.database
