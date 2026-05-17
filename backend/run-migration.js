require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { Pool } = require('pg');

const pool = new Pool({ connectionString: process.env.DATABASE_URL });

async function runMigration() {
  try {
    console.log('[MIGRATION] Starting...');
    const migrationPath = path.join(__dirname, 'migrate.sql');
    const sql = fs.readFileSync(migrationPath, 'utf8');
    
    // Split by semicolon and filter empty statements
    // Remove comment lines first, then split
    const lines = sql.split('\n')
      .filter(line => !line.trim().startsWith('--'))
      .join('\n');
    
    const statements = lines.split(';')
      .map(s => s.trim())
      .filter(s => s && s.length > 0);
    
    console.log('[MIGRATION] Found', statements.length, 'statements');
    
    for (let i = 0; i < statements.length; i++) {
      const stmt = statements[i];
      console.log(`[MIGRATION] ${i+1}/${statements.length}: ${stmt.substring(0, 50)}...`);
      try {
        await pool.query(stmt);
      } catch (err) {
        console.error(`[MIGRATION] ERROR on statement ${i+1}:`, err.message);
        throw err;
      }
    }
    
    console.log('[MIGRATION] ✓ Completed successfully!');
    await pool.end();
    process.exit(0);
  } catch (err) {
    console.error('[MIGRATION] ✗ Error:', err.message);
    console.error(err);
    await pool.end();
    process.exit(1);
  }
}

runMigration();
